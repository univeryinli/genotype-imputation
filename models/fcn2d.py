#!/usr/bin/python
# -*- coding: iso-8859-15 -*-
import os,json,torch,time,sys,copy,math,random
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import torch.optim as optim
import numpy as np
import pandas as pd


class Gene(nn.Module):
    def __init__(self, channel_in,channel_out):
        super(Gene, self).__init__()
        self.pad=nn.ZeroPad2d((0,0,0,996))

        self.conv1 = nn.Sequential(
            nn.Conv2d(channel_in, 4, (5, 1), stride=(3, 1), padding=0),
            nn.Conv2d(4, 4, (3, 1), stride=1, padding=0),
            nn.Conv2d(4,8,(5,1),stride=(3,1),padding=0),
            nn.Conv2d(8, 8, (3, 1), stride=1, padding=0),
        )

        self.conv2=nn.Sequential(
            nn.Conv2d(8,32,(5, 1), stride=(3, 1), padding=0),
            nn.Conv2d(32, 32, (3, 1), stride=1, padding=0),
            nn.Conv2d(32,64,(5,1),stride=(3,1),padding=0),
            nn.Conv2d(64, 64, (3, 1), stride=1, padding=0),
        )
        self.conv3=nn.Sequential(
            nn.Conv2d(64,128,(5, 1), stride=(3, 1), padding=0),
            nn.Conv2d(128, 128, (3, 1), stride=1, padding=0),
            nn.Conv2d(128,256,(5,1),stride=(3,1),padding=0),
            nn.Conv2d(256, 256, (3, 1), stride=1, padding=0)
        )

        self.cnn1=nn.Conv2d(1,4,(3,3),stride=1,padding=(0,262))

        self.conv4=nn.Sequential(
            nn.Conv2d(4,16,(5, 5), stride=(3,3), padding=0),
            nn.Conv2d(16, 16, (3, 3), stride=1, padding=0),
            nn.Conv2d(16,64,(5,5),stride=(3,3),padding=0),
            nn.Conv2d(64, 64, (3, 3), stride=1, padding=0)
        )

        self.conv5=nn.Sequential(
            nn.Conv2d(64,256,(5, 5), stride=(3, 3), padding=0),
            nn.Conv2d(256, 256, (3, 3), stride=1, padding=0),
            nn.Conv2d(256,channel_out,(5,5),stride=(3,3),padding=0),
            nn.Conv2d(channel_out, channel_out, (3, 3), stride=1, padding=0)
        )
        self.pool=nn.MaxPool2d((8,8))
        self.linear=nn.Linear(512,200)
    def forward(self, x1,x2):
        x1=self.pad(x1)
        x1=self.conv1(x1)
        x1=nn.functional.relu(x1,inplace=True)
        x1=self.conv2(x1)
        x1=nn.functional.relu(x1,inplace=True)
        x1=self.conv3(x1)
        x1=nn.functional.relu(x1,inplace=True)
        x1=x1.view(x1.shape[0],1,-1,500)
        x2=x2.view(x2.shape[0],1,1,500)
        merge=x1*x2
        merge=self.cnn1(merge)
        merge=nn.functional.relu(merge,inplace=True)
        merge=self.conv4(merge)
        merge=nn.functional.relu(merge,inplace=True)
        merge=self.conv5(merge)
        merge=nn.functional.relu(merge,inplace=True)
        merge=self.pool(merge)

        merge=merge.view(merge.shape[0], -1)

        merge=self.linear(merge)
        merge=nn.functional.relu(merge,inplace=True)
        return merge