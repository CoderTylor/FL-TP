#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import scipy.signal

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class LocalUpdate(object):
    def __init__(self, args, Train_dataset, Trudataset,idxs, logger):
        self.args = args
        self.logger = logger
        self.Train_dataset=Train_dataset
        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            Trudataset, list(idxs))
        self.device = 'cuda' if args.gpu else 'cpu'
        # Default criterion set to NLL loss function
        # self.criterion = nn.NLLLoss().to(self.device)
        self.criterion =nn.MSELoss(reduction='sum')

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True,drop_last=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=int(len(idxs_val)/10), shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=int(len(idxs_test)/10), shuffle=False)
        return trainloader, validloader, testloader

    def update_weights(self, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []
        Total,TP,FP=0.00,0.00,0.00

        # # 归一化数据x的
        # std = self.Train_dataset.std.repeat(BatchSize, 10, 1)
        # # std = std[:, :, 2:5].to(self.device)
        # mn = self.Train_dataset.mn.repeat(BatchSize, 10, 1)
        # # mn = mn[:, :, 2:5].to(self.device)
        # rg = self.Train_dataset.range.repeat(BatchSize, 10, 1)
        # # rg = rg[:, :, 2:5].to(self.device)
        # 归一化数据y的
        BatchSize=128

        mn_Y = self.Train_dataset.mn_Y.repeat(BatchSize, 10, 1)
        # 求6个输入值的范围,全部都做归一化（是否要归一化attacktype这次先没有）
        range_Y = self.Train_dataset.range_Y.repeat(BatchSize, 10, 1)
        # 求6个归一化范围
        std_Y = self.Train_dataset.std_Y.repeat(BatchSize, 10, 1)


        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                images = images.to(torch.float32)
                labels = labels.to(torch.float32)
                model.zero_grad()
                log_probs = model(images)
                # print(images[0],"images")
                # print(log_probs[0],"log_probs")
                loss = self.criterion(log_probs, labels)
                # loss = torch.clip(loss, -10000,10000)
                loss.backward()
                optimizer.step()

                log_probs=log_probs.detach().cpu()
                log_probs = (log_probs * (range_Y * std_Y) + mn_Y)
                labels=labels.detach().cpu()
                labels = (labels * (range_Y * std_Y) + mn_Y)
                #
                # print(log_probs,"log_probs")
                #
                # print(labels,"labels")

                attackPredictons = np.around(log_probs[:, :, 2])

                all=attackPredictons.shape[0]*attackPredictons.shape[1]
                # attack_0_n = np.count_nonzero(attackPredicton)
                if len(np.nonzero(attackPredictons))==0:
                    attack_number = 0
                else:
                    # attack_number=len(np.nonzero(attackPredictons)[0])
                    # print(np.nonzero(attackPredictons)[0])
                    attack_number=np.nonzero(attackPredictons).shape[0]
                # print(np.nonzero(attackPredictons).shape[0])
                non_attack_number=all-attack_number
                Total+=all
                TP+=attack_number
                FP+=non_attack_number


                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))

                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        # print(Total)
        # print(FP)
        # print(FP / Total, "准确率")
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), FP/Total

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """
        predictionError=[]
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        BatchSize=128
        TP = 0
        FP = 0
        Total = 0
        for batch_idx, (images, labels) in enumerate(self.testloader):
            # if images.shape[0] != BatchSize:
            #     continue
            # print(images.shape,"132456434444")
            BatchSize=images.shape[0]
            local_error=0
            images, labels = images.to(self.device), labels.to(self.device)
            images = images.to(torch.float32)
            local_labels = labels.to(torch.float32)
            # Inference
            # print("images",images)

            predY = model(images)
            batch_loss = self.criterion(predY, local_labels)
            loss += batch_loss.item()
            # print('predY',predY)
            # # Prediction
            # _, pred_labels = torch.max(outputs, 1)
            # pred_labels = pred_labels.view(-1)
            # correct += torch.sum(torch.eq(pred_labels, labels)).item()
            # total += len(labels)

            std = self.Train_dataset.std.repeat(BatchSize, 10, 1)
            std = std[:, :,2:5].to(self.device)
            mn = self.Train_dataset.mn.repeat(BatchSize, 10, 1)
            mn = mn[:, :,2:5].to(self.device)
            rg = self.Train_dataset.range.repeat(BatchSize, 10, 1)
            rg = rg[:, :,2:5].to(self.device)
            # print("predY",predY)
            predY[:, :,0:2] = (predY[:, :,0:2] * (rg[:, :,2:4] * std[:, :,2:4]) + mn[:, :,2:4])
            predY =predY.detach().cpu()
            pY = np.array(predY)
            # pY = scipy.signal.savgol_filter(pY, window_length=5, polyorder=2, axis=1)
            # local_labels = (local_labels * (rg * std) + mn).detach().cpu()
            Y=local_labels
            Y[:, :,0:2] = (Y[:, :,0:2] * (rg[:, :,2:4] * std[:, :,2:4]) + mn[:, :,2:4])
            Y=Y.detach().cpu()
            Y = np.array(Y)

            # pY[:, :-10, :] = Y[:, :-10, :]
            # error = ((Y[:, :, 0] - pY[:, :, 0]) ** 2 + (Y[:, :, 1] - pY[:, :, 1]) ** 2) ** 0.5
            # # predictionError.append(error.sum(0))
            # print(error.sum(0))


            for i in range(BatchSize):
                error = ((Y[i, -5:, 0] - pY[i, -5:, 0]) ** 2 + (Y[i, -5:, 1] - pY[i, -5:, 1]) ** 2) ** 0.5
                error = error.sum(0) / 5
                local_error += error
                # local_error += error.sum(0)
                # print(pY[i, :, 2])
                Attack_Prediction=np.around(predY[i, :, 2])
                Attack_Prediction_Error=Attack_Prediction-Y[i, :, 2]
                # print("predY[i, :, 2]",predY[i, :, 2])
                # print("Y[i, :, 2]",Y[i, :, 2])


                cnt_array = np.where(Attack_Prediction_Error, 0, 1)
                TP+=np.sum(cnt_array)
                # print(Attack_Prediction_Error.shape)
                FP+=Attack_Prediction_Error.shape[0]-TP
                Total+=Attack_Prediction_Error.shape[0]

            local_error /= BatchSize
            predictionError.append(local_error)
        # print(sum(predictionError))
        meterError = sum(predictionError) / len(predictionError)
        accuracy = TP/Total
        return meterError, loss, accuracy


def test_inference(args, model, test_dataset,train_dataset):
    """ Returns the test accuracy and loss.
    """
    BatchSize=128
    predictionError=[]
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    TP = 0.00
    FP = 0
    Total = 0.00
    device = 'cuda' if args.gpu else 'cpu'
    # criterion = nn.NLLLoss().to(device)
    criterion = nn.MSELoss(reduction='sum').to(device)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False,drop_last=True)

    mn_Y = train_dataset.mn_Y.repeat(BatchSize, 10, 1).detach().cpu()
    # 求6个输入值的范围,全部都做归一化（是否要归一化attacktype这次先没有）
    range_Y = train_dataset.range_Y.repeat(BatchSize, 10, 1).detach().cpu()
    # 求6个归一化范围
    std_Y = train_dataset.std_Y.repeat(BatchSize, 10, 1).detach().cpu()

    # print(mn_Y,mn_Y,std_Y)

    for batch_idx, (images, labels) in enumerate(testloader):
        # if images.shape[0] != BatchSize:
        #     continue
        BatchSize=images.shape[0]
        local_error=0
        local_error_count=0
        images = images.to(torch.float32)
        labels = labels.to(torch.float32)
        images, local_labels = images.to(device), labels.to(device)

        # Inference
        predY = model(images)

        # print("images",images[0])
        # print("local_labels",local_labels)
        # with open("log.txt", "a") as f:
        #     f.write("local_labels"+str(local_labels)+"\n")  # 自带文件关闭功能，不需要再写f.close()
        #     f.write("predY"+str(predY)+"\n")  # 自带文件关闭功能，不需要再写f.close()

        batch_loss = criterion(predY, local_labels)
        loss += batch_loss.item()

        predY = predY.detach().cpu()

        predY=(predY * (range_Y * std_Y) + mn_Y)
        # predY[:, :, 0:2] = (predY[:, :, 0:2] * (rg[:, :, 2:4] * std[:, :, 2:4]) + mn[:, :, 2:4])

        pY = np.array(predY)
        local_labels=local_labels.detach().cpu()
        # pY =  scipy.signal.savgol_filter(pY, window_length=5, polyorder=2,axis=1)
        # local_labels = (local_labels*(rg*std)+mn).detach().cpu()
        local_labels=(local_labels * (range_Y * std_Y) + mn_Y)

        # local_labels[:, :, 0:2] = (local_labels[:, :, 0:2]*(rg[:, :, 2:4]*std[:, :, 2:4])+mn[:, :, 2:4])
        # local_labels=local_labels.detach().cpu()
        Y = np.array(local_labels)
        # for ys in Y:
        #     print("111",ys)

        # print(labels[0],"label[0]")

        # pY[:,:-10,:] = Y[:,:-10,:]
        # print(pY.shape,"sssssssssssssssss")
        # error = ((Y[:, :, 0] - pY[:, :, 0]) ** 2 + (Y[:, :, 1] - pY[:, :, 1]) ** 2) ** 0.5
        # # predictionError.append(error.sum(0))
        # print(error.sum(0))
        # with open("log.txt", "a") as f:
        #     f.write("Calculated_local_labels"+str(local_labels)+"\n")  # 自带文件关闭功能，不需要再写f.close()
        #     f.write("Calculated_predY"+str(predY)+"\n")  # 自带文件关闭功能，不需要再写f.close()

        for i in range(BatchSize):
            error = ((Y[i, -5:, 0] - pY[i, -5:, 0]) ** 2 + (Y[i, -5:, 1] - pY[i, -5:, 1]) ** 2) ** 0.5
            error=error.sum(0)/5
            # print(Y[i, -5:, :],"1111")
            # print(pY[i, -5:, :],"2222")
            # print(error)

            if round(Y[i][0][2]) == 0:
                # print(Attack_Prediction)
                local_error += error
                local_error_count+=1

            # print("predY[i, :, 2]",predY[i, :, 2])

            Attack_Prediction = np.around(predY[i, :, 2])
            Attack_Prediction_Error = np.around(Attack_Prediction - Y[i, :, 2])
            # print("Attack_Prediction",Attack_Prediction)
            # print("Attack_Prediction_Error",Attack_Prediction_Error)
            # print("Y[i, :, 2]", Y[i, :, 2])
            cnt_array = np.where(Attack_Prediction_Error, 0, 1)
            TP += np.sum(cnt_array)
            FP += (Attack_Prediction_Error.shape[0] - TP)
            Total += Attack_Prediction_Error.shape[0]
        local_error = local_error / (local_error_count)
        predictionError.append(local_error)
        # if round(Y[0][0][2])==0:
        #     # print(Attack_Prediction)
        #     local_error=local_error/BatchSize
        #     predictionError.append(local_error)
        # else:
        #     print(Attack_Prediction)
    # print(sum(predictionError))
    meterError = sum(predictionError) / len(predictionError)
    accuracy=TP/Total
        # # Prediction
        # _, pred_labels = torch.max(outputs, 1)
        # pred_labels = pred_labels.view(-1)
        # correct += torch.sum(torch.eq(pred_labels, labels)).item()
        # total += len(labels)

    # accuracy = correct/total
    return meterError, loss,accuracy
