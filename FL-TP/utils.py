#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid,VeRemi_iid
from torch.utils.data import Dataset, DataLoader
import os.path
from os import path
import pandas as pd
import random
import numpy as np
import scipy.signal
import pickle


class TrajectoryDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file='allData.csv'):
        """
            Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
            on a sample.
            """
        self.csv_file = csv_file
        # store X as a list, each element is a 100*42(len * attributes num) np array [velx;vely;x;y;acc;angle] * 7
        self.X_frames = []
        # store Y as a list, each element is a 100*4(len * attributes num) np array[velx;vely;x;y]
        self.Y_frames = []
        self.load_data()
        self.normalize_data()

    def __len__(self):
        return len(self.X_frames)

    def __getitem__(self, idx):
        single_data = self.X_frames[idx]
        single_label = self.Y_frames[idx]
        return (single_data, single_label)

    def load_data(self):
        dataS = pd.read_csv(self.csv_file)
        max_vehiclenum = np.max(dataS.vehicle_ID.unique())
        for vid in dataS.vehicle_ID.unique():
            print('{0} and {1}'.format(vid, max_vehiclenum))
            frame_ori = dataS[dataS.vehicle_ID == vid]
            frame = frame_ori[['pos_x', 'pos_y','attackerType','spd_x','spd_y', 'disChange_X', 'disChange_Y','time',
                               'RSSI']]
            # frame = frame_ori[['pos_x', 'pos_y','spd_x','spd_y','disChange_X', 'disChange_Y','time',
            #                    'RSSI','attackerType']]
            frame = np.asarray(frame)
            #            print(frame.shape[0])
            if frame.shape[0] < 5:
                continue
            # print(frame.shape,"sadsad")

            # frame[np.where(frame>4000)] = 0 # assign all 5000 to 0
            # remove anomalies, which has a discontinuious local x or local y
            dis = frame[1:, :2] - frame[:-1, :2]
            # print(dis.shape)
            dis = np.sqrt(np.power(dis[:, 0], 2) + np.power(dis[:, 1], 2))
            # print(dis.shape)
            idx = np.where(dis > 200)
            if not (idx[0].all):
                continue
            # split into several frames each frame have a total length of 100, drop sequence smaller than 130
            if (frame.shape[0] < 13):
                continue
            # frame = frame_ori[['pos_x', 'pos_y','attackerType', 'spd_x', 'spd_y','time','disChange_X','disChange_Y','spdChange_X','spdChange_Y','RSSI']]
            X = np.concatenate((frame[:-5, :2], frame[:-5, 3:]), axis=1)
            Y = (frame[2:, :3])

            count = 0
            for i in range(X.shape[0] - 10):
                if random.random() > 0.2:
                    continue
                j = i - 1
                if count > 2:
                    break
                # print('X[] shape',X[i:i+100,:].shape)
                self.X_frames = self.X_frames + [X[i:i + 10, :]]
                self.Y_frames = self.Y_frames + [Y[i:i + 10, :]]
                count = count + 1

    def normalize_data(self):
        # 对x归一化
        A = [list(x) for x in zip(*(self.X_frames))]
        A = torch.tensor(A)
        print('A:', A.shape)
        A = A.view(-1, A.shape[2])
        # 求6个输入值的中位数
        self.mn = torch.mean(A, dim=0)
        # 求6个输入值的范围,attacktype不需要归一化
        self.range = (torch.max(A, dim=0).values - torch.min(A, dim=0).values)
        # 求6个归一化范围
        self.range = torch.ones(self.range.shape, dtype=torch.double)
        self.std = torch.std(A, dim=0)
        # print(self.mn,self.std,self.range)
        self.X_frames = [(torch.tensor(item) - self.mn) / (self.std * self.range) for item in self.X_frames]

        # 对Y归一化
        A_Y = [list(y) for y in zip(*(self.Y_frames))]
        A_Y = torch.tensor(A_Y)
        A_Y = A_Y.view(-1, A_Y.shape[2])
        # 求3个输入值的中位数
        self.mn_Y = torch.mean(A_Y, dim=0)
        # 求6个输入值的范围,全部都做归一化（是否要归一化attacktype这次先没有）
        self.range_Y = (torch.max(A_Y, dim=0).values - torch.min(A_Y, dim=0).values)
        # 求6个归一化范围
        self.range_Y = torch.ones(self.range_Y.shape, dtype=torch.double)
        self.std_Y = torch.std(A_Y, dim=0)
        # print(self.Y_frames[0],"previsou")
        # 对y归一化
        self.Y_frames = [(torch.tensor(item) - self.mn_Y) / (self.std_Y * self.range_Y) for item in self.Y_frames]
        # print(self.Y_frames[0],"after")


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    if path.exists("my_dataset.pickle"):
        with open('my_dataset.pickle', 'rb') as data:
            dataset = pickle.load(data)
    else:
        dataset = TrajectoryDataset()
        with open('my_dataset.pickle', 'wb') as output:
            pickle.dump(dataset, output)

    #split dataset into train test and validation 7:2:1
    num_train = (int)(dataset.__len__()*0.7)
    num_test = (int)(dataset.__len__()*0.9) - num_train
    num_validation = (int)(dataset.__len__()-num_test-num_train)
    trainDataset, testDataset, validationDataset = torch.utils.data.random_split(dataset, [num_train, num_test, num_validation])
    # print(trainDataset.__len__())
    # print(dataset.__len__())
    user_groups = VeRemi_iid(trainDataset, 20,dataset)
    # user_groups = VeRemi_iid(trainDataset, args.num_users)

    return trainDataset, testDataset, user_groups, dataset

def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def custom_weights(parameters,weights):
    """
    Returns the custom of the model.
    """
    # print(weights)
    # print(len(weights))

    sum_weights=sum(weights)
    average_weights=sum_weights/len(weights)
    for i in range(len(weights)):
        weights[i]=weights[i]/average_weights
    parameters_custom = copy.deepcopy(parameters[0])
    # print(weights[0])
    # parameters_custom=parameters_custom*weights[0]
    for key in parameters_custom.keys():
        # print(parameters_custom[key])
        parameters_custom[key] = (parameters[0][key] * weights[0])
    for key in parameters_custom.keys():
        for i in range(1, len(parameters)):
            parameters_custom[key] += (parameters[i][key]*weights[i])
        parameters_custom[key] = torch.div(parameters_custom[key], len(parameters))
    return parameters_custom

def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return
