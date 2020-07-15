#coding=GBK
import os,json,torch,time,sys,copy,math,random
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import torch.optim as optim
import numpy as np


class convbn(nn.Module):
    def __init__(self,channel_in, channel_out,kernel_size,stride=1,padding=0):
        super(convbn,self).__init__()
        self.convbn = nn.Sequential(nn.Conv1d(channel_in, channel_out, kernel_size, stride=stride,padding=padding),
                                     nn.ReLU(inplace=True),
                                     nn.BatchNorm1d(channel_out))

    def forward(self, x):
        return self.convbn(x)


class Gene(nn.Module):
    def __init__(self):
        super(Gene, self).__init__()
        self.conv1 = nn.Sequential(convbn(1, 8, 3, stride=2),
                                   convbn(8, 8, 3))
        self.pool = nn.MaxPool1d(3, stride=2)
        self.conv1_1 = nn.Sequential(convbn(8, 8, 3, padding=1),
                                     convbn(8, 8, 3, padding=1))
        self.conv1_2 = nn.Sequential(convbn(8, 8, 3, padding=1),
                                     convbn(8, 8, 3, padding=1))
        self.conv1_3 = nn.Sequential(convbn(8, 8, 3, padding=1),
                                     convbn(8, 8, 3, padding=1))
        self.conv2 = nn.Sequential(convbn(8, 16, 3, stride=2),
                                   convbn(16, 16, 3))
        self.conv2_1 = nn.Sequential(convbn(16, 16, 3, padding=1),
                                     convbn(16, 16, 3, padding=1))
        self.conv2_2 = nn.Sequential(convbn(16, 16, 3, padding=1),
                                     convbn(16, 16, 3, padding=1))
        self.conv2_3 = nn.Sequential(convbn(16, 16, 3, padding=1),
                                     convbn(16, 16, 3, padding=1))
        self.conv3 = nn.Sequential(convbn(16, 32, 3, stride=2),
                                   convbn(32, 32, 3))
        self.conv3_1 = nn.Sequential(convbn(32, 32, 3, padding=1),
                                     convbn(32, 32, 3, padding=1))
        self.conv3_2 = nn.Sequential(convbn(32, 32, 3, padding=1),
                                     convbn(32, 32, 3, padding=1))
        self.conv3_3 = nn.Sequential(convbn(32, 32, 3, padding=1),
                                     convbn(32, 32, 3, padding=1))
        self.conv4 = nn.Sequential(convbn(32, 64, 3, stride=2),
                                   convbn(64, 64, 3))
        self.conv4_1 = nn.Sequential(convbn(64, 64, 3, padding=1),
                                     convbn(64, 64, 3, padding=1))
        self.conv4_2 = nn.Sequential(convbn(64, 64, 3, padding=1),
                                     convbn(64, 64, 3, padding=1))
        self.conv5 = nn.Sequential(convbn(64, 64, 3, stride=2),
                                   convbn(64, 64, 3))
        self.conv5_1 = nn.Sequential(convbn(64, 64, 3, padding=1),
                                     convbn(64, 64, 3, padding=1))
        self.conv5_2 = nn.Sequential(convbn(64, 64, 3, padding=1),
                                     convbn(64, 64, 3, padding=1))
        self.pool2 = nn.MaxPool1d(10)
        self.linear = nn.Linear(64, 1000)

    def forward(self, x1):
        x1 = self.conv1(x1)
        x1 = self.pool(x1)
        x2 = self.conv1_1(x1)
        x2 = x1 + x2
        x1 = self.conv1_2(x2)
        x1 = x2 + x1
        x2 = self.conv1_3(x1)
        x1 = self.conv2(x2)
        x2 = self.conv2_1(x1)
        x2 = x2 + x1
        x1 = self.conv2_2(x2)
        x1 = x1 + x2
        x2 = self.conv2_3(x1)
        x1 = self.conv3(x2)
        x2 = self.conv3_1(x1)
        x2 = x1 + x2
        x1 = self.conv3_2(x2)
        x1 = x2 + x1
        x2 = self.conv3_3(x1)
        x1 = self.conv4(x2)
        x2 = self.conv4_1(x1)
        x2 = x2 + x1
        x1 = self.conv4_2(x2)
        x2 = self.conv5(x1)
        x1 = self.conv5_1(x2)
        x1 = x2 + x1
        x2 = self.conv5_2(x1)
        x2 = self.pool2(x2)
        x2 = x2.squeeze(2)
        x2 = self.linear(x2)
        return x2