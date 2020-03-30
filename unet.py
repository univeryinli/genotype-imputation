#!/usr/bin/python
# -*- coding: iso-8859-15 -*-
import os,json,torch,time,sys,copy,math,random
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd


class TrainData(Dataset):
    def __init__(self,input_data,window_size,mask):
        self.input_data=input_data
        self.window_size=window_size
        self.step=window_size//8
        self.window_matrix = [i for i in range(0, self.input_data.shape[0] - window_size, self.step)]
        self.people_index = random.sample([i for i in range(input_data.shape[1])], int((0.1*input_data.shape[1])))
        self.all_mask=mask.random_single_mask(input_data,self.people_index)
        print('train ample is:' + str(len(self.window_matrix)))

    def __len__(self):
        return len(self.window_matrix)

    def __getitem__(self, index):
        dic = {}
        ix = self.window_matrix[index]
        encode_input = torch.from_numpy(self.input_data[ix:ix + self.window_size-1])
        encode_mask=self.all_mask[ix:ix + self.window_size-1]
        encode_target = encode_input
        encode_input = encode_input.masked_fill(encode_mask, -1)
        dic['encode_input'] = encode_input.unsqueeze(0)
        dic['encode_mask'] = encode_mask.unsqueeze(0)
        dic['encode_target']=encode_target.unsqueeze(0)
        return dic


class TestData(Dataset):
    def __init__(self,input_data,window_size,mask):
        self.input_data=input_data
        self.window_size=window_size
        self.step=window_size//8
        self.window_matrix = [i for i in range(0, self.input_data.shape[0] - window_size, self.step)]
        self.people_index = random.sample([i for i in range(input_data.shape[1])], int((0.1*input_data.shape[1])))
        self.all_mask = mask.random_single_mask(input_data, self.people_index)
        print('test ample is:' + str(len(self.window_matrix)))

    def __len__(self):
        return len(self.window_matrix)

    def __getitem__(self, index):
        dic = {}
        ix = self.window_matrix[index]
        encode_input = torch.from_numpy(self.input_data[ix:ix + self.window_size-1])
        encode_mask = self.all_mask[ix:ix + self.window_size - 1]
        encode_target = encode_input
        encode_input = encode_input.masked_fill(encode_mask, -1)
        dic['encode_input'] = encode_input.unsqueeze(0)
        dic['encode_mask'] = encode_mask.unsqueeze(0)
        dic['encode_target']=encode_target.unsqueeze(0)
        return dic


class PreData(Dataset):
    def __init__(self,input_data,window_size,mask):
        self.window_size=window_size
        self.step=int(window_size*0.6)
        self.window_matrix = [i for i in range(0, input_data.shape[0], self.step)]
        self.mask= mask
        #random mask predict
        self.encode_mask=mask.random_mask(input_data)
        # gene_chip mask predict

        self.input_data=torch.from_numpy(input_data).masked_fill(self.encode_mask, -1)
        print('test ample is:' + str(len(self.window_matrix)))

    def __len__(self):
        return len(self.window_matrix)

    def __getitem__(self, index):

        ix = self.window_matrix[index]
        if ix>=(self.input_data.shape[0]-self.step):
            encode_input=self.input_data[ix:self.input_data.shape[0]]
        else:
            encode_input = self.input_data[ix:ix + self.window_size]

        return encode_input.unsqueeze(0).unsqueeze(0)


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d((3,3),stride=(3,3)),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=(3,3), mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class Gene(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(Gene, self).__init__()
        self.n_channels=n_channels
        self.n_class=n_classes
        self.inc = inconv(n_channels, 32)
        self.down1 = down(32, 64)
        self.down2 = down(64, 128)
        self.down3 = down(128, 256)
        self.down4 = down(256, 256)
        self.up1 = up(512, 128)
        self.up2 = up(256, 64)
        self.up3 = up(128, 32)
        self.up4 = up(64, 32)
        self.outc = outconv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x4 = self.up1(x5, x4)
        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return torch.nn.functional.sigmoid(x)


'''
class TrainModel():
    def __init__(self,model,dataloads,scheduler,board=None,):
        self.model=model
        self.dataloads=dataloads
        self.scheduler=scheduler
        self.board=board
        self.num_epochs=35
        self.train_step
'''


def predict():
    input_data = np.load('processed_hapmap/hapmap3.impute.hap.npy')
    input_data = pd.DataFrame(input_data)

    masked = np.load('processed/mask.npy')

    window_size = 1000
    length = input_data.shape[0]
    col = [i for i in range(input_data.shape[1]) if i % 2 == 0]

    index_file_n = open('processed/index_file.txt', 'r')
    index_file=[eval(line) for line in index_file_n]
    index_file_n.close()

    val_dataset = PreData(input_data[index_file[0][1]], window_size, masked)
    #val_dataset = PreData(input_data, window_size, masked)

    gene = Gene(1, 1)
    model=torch.load('runs/5.best_model_wts')
    model=model['state']
    gene.load_state_dict(model)
    #gene=gene.cuda(1)

    import time
    start=time.time()
    for i in range(len(val_dataset)):
        #cal=gene(val_dataset[i].float().cuda(1))[0][0].data.cpu()
        cal = gene(val_dataset[i].float())[0][0].data
        if i==0:
            result=cal[0:int(window_size*0.8)]
        elif i==len(val_dataset)-1:
            result=cal[int(0.2*window_size):cal.shape[0]]
        else:
            result=cal[int(0.2*window_size):int(window_size*0.8)]
        #result=result+cal[int(0.2*window_size):int(window_size*0.8)].tolist()
        print(i)
        #if i==2000:
        torch.save(result,os.path.join('./result_test',str(i)))
    #    break
    stop=time.time()
    time_use=stop-start
    print(time_use)


def predict_new():
    input_data = np.load('processed_hapmap/hapmap3.impute.hap.1440000.npy')
    input_data = pd.DataFrame(input_data)

    masked = np.load('processed_hapmap/mask1440000.npy')

    window_size = 1000

    val_dataset = PreData_new(input_data, window_size, masked)
    #val_dataset = PreData(input_data, window_size, masked)

    gene = Gene(1, 1)
    model=torch.load('runs/5.best_model_wts')
    model=model['state']
    gene.load_state_dict(model)
    gene=gene.cuda()

    import time
    start=time.time()
    for i in range(len(val_dataset)):
        cal=gene(val_dataset[i].float().cuda())[0][0].data
        #cal = gene(val_dataset[i].float())[0][0].data
        if i==0:
            result=cal[0:int(window_size*0.8)]
        elif i==len(val_dataset)-1:
            result=cal[int(0.2*window_size):cal.shape[0]]
        else:
            result=cal[int(0.2*window_size):int(window_size*0.8)]
        #result=result+cal[int(0.2*window_size):int(window_size*0.8)].tolist()
        print(i)
        #if i==2000:
        torch.save(result,os.path.join('./result_test',str(i)))
    stop=time.time()
    time_use=stop-start
    print(time_use)