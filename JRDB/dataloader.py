import torch, torchvision
from torchvision import transforms
import json, os 
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import pickle, random, copy
# from .noisytraj.imu_cal import imu_cal
import random


class Dataset_JRDB(Dataset):
    def __init__(self, C, flag='train', shuffle_unsup = False):

        super().__init__()

        assert flag in ['train', 'test', 'valid']
        self.flag = flag
        self.shuffle_unsup = shuffle_unsup
        self.C = C
        with open(os.path.join( "/home/haichao/code/gps/gpspred/JRDB/jrdb.pkl"), 'rb') as f:
            sensor_data, gt = pickle.load(f)
        print("load data from pkl...")
        self.transform = transforms.Compose(
                        [
                            transforms.ToTensor(),
                            # transforms.Normalize((0.5), (0.5))
                        ])   
        data_2D, data_3D  = torch.Tensor(sensor_data), torch.Tensor(gt)
        self.bbx4, self.imu19, self.nearst_gps, self.interl_gps  =  data_2D, data_3D, data_3D, data_3D # sensor_data_3D, gt_4d
        # video, timestamp, self.bbx4, self.imu19, self.nearst_gps, self.interl_gps = data["video"], data["timestamp"], data["bbx"], data["imu"], data["nearst_gps"], data["interl_gps"]

        # self.__load_data__()
        # self.transform2 = transforms.Compose(
        #                 [transforms.Normalize((0.5), (0.5))])   


        # def norm(a):
        #     channel = a.shape[0]
        #     return transforms.Normalize((0.5,)*channel, (0.5,)*channel)(a).permute(1,2,0)
        # self.interl_gps = norm(self.transform(self.interl_gps)/100.)
        # self.nearst_gps = norm(self.transform(self.nearst_gps)/100.)#.permute(1,2,0))
        # self.imu19 = norm(self.transform(self.imu19)/64.)#.permute(1,2,0))
        # self.bbx4 = norm(self.transform(self.bbx4)/2048.)#.permute(1,2,0))
        self.interl_gps=[ i/100. for i in self.interl_gps    if i.shape[1]==3 ] # TODO: now only support 3 pedestrians, 2 pedestrians will be supported in the future
        self.nearst_gps=[i /100. for i in self.nearst_gps    if i.shape[1]==3 ] # TODO: needs a better way to handle the dimension in model and dataset
        self.imu19=[i/64. for i in self.imu19                if i.shape[1]==3 ]
        self.bbx4=[i/2048. for i in self.bbx4                if i.shape[1]==3 ]

        # split_point = 2500
        split_point = int(len(self.interl_gps)*0.8)
        if flag=='valid':
            self.interl_gps = self.interl_gps[:split_point]
            self.nearst_gps = self.nearst_gps[:split_point]
            self.imu19 = self.imu19[:split_point]
            self.bbx4 = self.bbx4[:split_point]
        elif flag=='train':
            self.interl_gps = self.interl_gps[split_point:]
            self.nearst_gps = self.nearst_gps[split_point:]
            self.imu19 = self.imu19[split_point:]
            self.bbx4 = self.bbx4[split_point:]
        else:
            raise ValueError("flag")
        self.length = len(self.interl_gps)

    def choose_a(self, a, index):
        interl_gps = self.interl_gps[index][:,a,:] # 200, 3, 2     [ timestamps, pedestrians, dimension(x,y)] pedestrians might be 2 or 3
        
        passersby_interl_gps = self.remove_dim_a(self.interl_gps[index], a)


        # noisy_traj = torch.cat([self.nearst_gps[index], self.imu19[index]])
        nearst_gps = self.nearst_gps[index] [:,a,:]

        passersby_nearst_gps = self.remove_dim_a(self.nearst_gps[index], a)

        imu19 = self.imu19[index][:,a,:]
        bbx4 = self.bbx4[index][:,a,:]
        passersby_bbx4 = self.remove_dim_a(self.bbx4[index], a)
        return interl_gps, nearst_gps, imu19, bbx4, passersby_bbx4, passersby_interl_gps, passersby_nearst_gps

    def __getitem__(self, index):
        a = random.choice(range(self.interl_gps[index].shape[1]) )  # randomly choose a pedestrian

        # interl_gps = self.interl_gps[index][:,a,:] # 200, 3, 2     [ timestamps, pedestrians, dimension(x,y)] pedestrians might be 2 or 3        
        # passersby_interl_gps = self.remove_dim_a(self.interl_gps[index], a)
        # # noisy_traj = torch.cat([self.nearst_gps[index], self.imu19[index]])
        # nearst_gps = self.nearst_gps[index] [:,a,:]
        # passersby_nearst_gps = self.remove_dim_a(self.nearst_gps[index], a)
        # imu19 = self.imu19[index][:,a,:]
        # bbx4 = self.bbx4[index][:,a,:]
        # passersby_bbx4 = self.remove_dim_a(self.bbx4[index], a)
        interl_gps, nearst_gps, imu19, bbx4, passersby_bbx4, passersby_interl_gps, passersby_nearst_gps = self.choose_a(a, index)
        passersby_bbx4, passersby_interl_gps, passersby_nearst_gps = passersby_bbx4, passersby_interl_gps, passersby_nearst_gps

        

        # return interl_gps, nearst_gps, imu19, bbx4, passersby_interl_gps, passersby_nearst_gps
        if self.shuffle_unsup:
            shuffle_pkg = [self.choose_a(0, index), self.choose_a(1, index), self.choose_a(2, index)]
            return interl_gps, nearst_gps, imu19, bbx4, passersby_bbx4, passersby_interl_gps, passersby_nearst_gps, shuffle_pkg
        else:
            return interl_gps, nearst_gps, imu19, bbx4, passersby_bbx4, passersby_interl_gps, passersby_nearst_gps
    
    def remove_dim_a(self,tensor_before, a):
        if a < tensor_before.size(1) - 1:
            tensor_after = torch.cat((tensor_before[:, :a, :], tensor_before[:, a+1:, :]), dim=1)
        else:
            # Handle the case where one of the slices becomes empty
            tensor_after = tensor_before[:, :a, :] if a > 0 else tensor_before[:, a+1:, :]
        return tensor_after

    def IMU_to_trajectory(self, ):
        # TODO: from IMU to Trajectory
        pass
    def __len__(self):
        return self.length

    def __load_data__(self,):
        pass
        print("interl_gps.shape:{}\nseq_in_FTM2.shape:{}\nimu19.shape:{}\nbbx4.shape:{}\n"
            .format(self.interl_gps.shape, self.nearst_gps.shape, self.imu19.shape, self.bbx4.shape))



def load_data_jrdb(C, datasetname=Dataset_JRDB, shuffle_unsup=False):
    Dataset_name = datasetname
    train_dataset = Dataset_name(C, flag='train', shuffle_unsup=shuffle_unsup)
    # train_dataloader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
    valid_dataset = Dataset_name(C, flag='valid')
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=6000, shuffle=False)
    # valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=64, shuffle=False)
    return train_dataloader, valid_dataloader



