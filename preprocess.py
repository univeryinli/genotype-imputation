import os,torch
import random as random
import numpy as np


class Data_div():
    def __init__(self,path):
        self.path=os.path.join(path,'index.txt')

    def data_div(self,input_list,rate):
        if rate<1:
            col = [i for i in input_list if i % 2 == 0]
            train_index = random.sample(col, int(rate * len(col)))
            val_index = list(set(col) - set(train_index))
            train_index = sum([[i, i + 1] for i in train_index], [])
            val_index = sum([[i, i + 1] for i in val_index], [])
            return (train_index,val_index)
        elif rate>=1:
            col = [i for i in input_list if i % 2 == 0]
            train_index = random.sample(col, rate)
            val_index = list(set(col) - set(train_index))
            train_index = sum([[i, i + 1] for i in train_index], [])
            val_index = sum([[i, i + 1] for i in val_index], [])
            return (train_index, val_index)

    def data_sample(self,rate):
        if rate<1:
            col = [i for i in input_list if i % 2 == 0]
            train_index = random.sample(col, int(rate * len(col)))
            return train_index
        elif rate>=1:
            col = [i for i in input_list if i % 2 == 0]
            train_index = random.sample(col, rate)
            train_index = sum([[i, i + 1] for i in train_index], [])
            return train_index
'''
    def data_index_save(self):
        temp=(self.train_index, self.val_index)
        index_file = open(self.path, 'w')
        for i in temp:
            index_file.write(str(i)+'\n')
        index_file.close()

    def data_index_load(self):
        index_file = open(self.path, 'r')
        self.train_index=eval(index_file.readline())
        self.val_index=eval(index_file.readline())
'''

class Mask():
    def __init__(self,gene_chip):
        self.gene_chip=(1-torch.from_numpy(gene_chip)).bool()
        self.missing_rate=0

    #random mask all people
    def random_mask(self,input_data):
        encode_mask = torch.rand(input_data.shape) > (1 - self.missing_rate)
        return encode_mask

    #index is size : [1,2,3],use chip to mask one person
    def chip_single_mask(self,input_data ,index,people_index):
        encode_mask=torch.zeros(input_data.shape,dtype=torch.bool)
        encode_mask[:,people_index]=self.gene_chip.view(-1,1)
        return encode_mask[:,index]

    #use random mask to mask one person
    def random_single_mask(self,input_data,index):
        random_single_mask=torch.zeros(input_data.shape,dtype=torch.bool)
        random_single_mask[:,index]=self.random_mask(random_single_mask[:,0]).view(-1,1)
        return random_single_mask

    #calculate the maf
    def maf_cal(self,input_data):
        temp = np.sum(input_data, axis=1) / (input_data.shape[1])
        temp=np.concatenate((temp.reshape(-1,1),(1-temp).reshape(-1,1)),axis=1)
        temp=np.min(temp,axis=1)
        self.maf_cal=torch.from_numpy(temp)

    # use maf mask to mask one person,index is size : [1,2,3],#maf is size [maf1,maf2]
    def maf_single_mask(self,input_data,index,maf):
        maf_mask = ((self.maf_cal <= maf[1]) & (self.maf_cal > maf[0]))
        maf_mask=(maf_mask & self.gene_chip)
        maf_single_mask = torch.zeros(input_data.shape,dtype=torch.bool)
        maf_single_mask[:,index] = maf_mask.view(-1,1)
        return maf_single_mask