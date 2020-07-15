#coding=GBK
import os,torch
import random as random
import numpy as np


class Data_Div():
    def sampler(self,input_list,rate):
        if rate<1:
            col = [i for i in input_list if i % 2 == 0]
            index = random.sample(col, int(rate * len(col)))
            index = sum([[i, i + 1] for i in index], [])
            return index
        elif rate>=1:
            col = [i for i in input_list if i % 2 == 0]
            index = random.sample(col, rate)
            index = sum([[i, i + 1] for i in index], [])
            return index


class Mask():
    def __init__(self,gene_chip):
        self.gene_chip=gene_chip
        self.missing_rate=0

    #random mask all people
    def random_mask(self,len):
        encode_mask = torch.rand(len) > (1 - self.missing_rate)
        return encode_mask

    #index is size : [1,2,3],use chip to mask one person
    def chip_single_mask(self,input_data ,people_index):
        return input_data[people_index,:].masked_fill(self.gene_chip.view(1,-1),1)

    def chip_mask(self,input_data):
        return input_data.masked_fill(self.gene_chip.view(-1,1),-1)

    #use random mask to mask one person
    def random_single_mask(self,input_data,index,people_index):
        random_single_mask=torch.zeros(input_data.shape,dtype=torch.bool)
        random_single_mask[:,people_index]=self.random_mask(random_single_mask[:,0]).view(-1,1)
        return random_single_mask[:,index]

    #calculate the maf
    def maf_cal(self,input_data):
        temp = torch.sum(input_data==1, dim=1).float() / (input_data.shape[1])
        temp=torch.cat((temp.view(-1,1),(1-temp).view(-1,1)),dim=1)
        temp=torch.min(temp,dim=1)[0]
        return temp

    # use maf mask to mask one person,index is size : [1,2,3],#maf is size [maf1,maf2]
    def maf_single_mask(self,input_data,index,maf):
        maf_mask = ((self.maf_cal <= maf[1]) & (self.maf_cal > maf[0]))
        maf_mask=(maf_mask & self.gene_chip)
        maf_single_mask = torch.zeros(input_data.shape,dtype=torch.bool)
        maf_single_mask[:,index] = maf_mask.view(-1,1)
        return maf_single_mask

    # 基于一定范围的maf进行随机mask，缺失率由初始缺失率决定
    def maf_random_mask(self,maf):
        maf_common = ((maf <= 0.5) & (maf > 0.05))
        mask=self.random_mask(maf.shape[0]) & maf_common
        return mask


