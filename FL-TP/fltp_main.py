#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar,LSTM_CNN
from utils import get_dataset, average_weights, exp_details,custom_weights
# from GC_datapre import *
import csv
import os.path
from os import path

if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)
    torch.cuda.set_device(0)
    device = 'cuda' if args.gpu else 'cpu'

    # # load dataset and user groups
    hidden_size = 256
    BatchSize = 128
    train_dataset, test_dataset, user_groups, dataset_parameters = get_dataset(args)
    train_loader_dataset = DataLoader(train_dataset, batch_size=BatchSize, shuffle=True)
    # Training_generator, Test, Valid, WholeSet,user_groups = get_dataloader(args)
    train_iter = iter(train_loader_dataset)
    x, y = train_iter.next()

    print(device)
    # BUILD MODEL， LSTM Trajectory Prediction
    if args.model == 'LSTM':
        global_model = LSTM_CNN(x.shape[2], y.shape[2], hidden_size, BatchSize)

    if path.exists("checkpoint.pth.tar"):
        global_model.load_state_dict(torch.load('checkpoint.pth.tar'))
    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # copy weights
    global_weights = global_model.state_dict()

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0
    accuracy=0

    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses ,weights = [], [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:
            local_model = LocalUpdate(args=args, Train_dataset=dataset_parameters,Trudataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            w, loss,weight = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            weights.append(copy.deepcopy(weight))

        # # update global weights
        # if accuracy < 0.85:
        #     global_weights = average_weights(local_weights)
        #
        # else:
        global_weights = custom_weights(local_weights,weights)

        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(args.num_users):
            local_model = LocalUpdate(args=args, Train_dataset=dataset_parameters,Trudataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            acc, loss,accuracy = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc)/len(list_acc))
        # Test inference after completion of training
        meterError, test_loss, accuracy = test_inference(args, global_model, test_dataset,dataset_parameters)
        # print()
        if (epoch+1) % 1 == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(test_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100*accuracy))
            print('meter_Error',meterError)
        # global_model
        torch.save(global_model.state_dict(), 'checkpoint.pth.tar')

        # 保存csv
        with open("add_weights_meterError.csv", "a") as csvfile:
            writer = csv.writer(csvfile)
            # 先写入columns_name
            writer.writerow([meterError,test_loss,accuracy])

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))


