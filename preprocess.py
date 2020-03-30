import os,torch
import random as random
import numpy as np


class Train_data_div():
    def __init__(self,input_data,div_rate,path):
        self.path=os.path.join(path,'index.txt')
        self.input_data=input_data
        if div_rate!=1:
            (self.train_index,self.val_index)=self.training_set_div(div_rate)
        else:
            self.train_index=[i for i in range(self.input_data.shape[1])]
            self.val_index=[i for i in range(self.input_data.shape[1])]

    def training_set_div(self,rate):
        col = [i for i in range(self.input_data.shape[1]) if i % 2 == 0]
        train_index = random.sample(col, int(rate*len(col)))
        val_index = list(set(col) - set(train_index))
        train_index = sum([[i, i + 1] for i in train_index], [])
        val_index = sum([[i, i + 1] for i in val_index], [])
        return (train_index,val_index)

    def training_index_save(self):
        temp=(self.train_index, self.val_index)
        index_file = open(self.path, 'w')
        for i in temp:
            index_file.write(str(i)+'\n')
        index_file.close()

    def training_index_load(self):
        index_file = open(self.path, 'r')
        self.train_index=eval(index_file.readline())
        self.val_index=eval(index_file.readline())

    def training_set_generation(self,path):
        train_set=np.load(path)
        np.save(path.strip('npy')+'train',train_set[self.train_index])
        np.save(path.strip('npy')+'val',train_set[self.val_index])


class Mask():
    def __init__(self,gene_chip):
        self.gene_chip=torch.from_numpy(gene_chip).bool()
        self.missing_rate=0

    #random mask all people
    def random_mask(self,input_data):
        encode_mask = torch.rand(input_data.shape) > (1 - self.missing_rate)
        return encode_mask

    #index is size : [1,2,3],use chip to mask one person
    def chip_single_mask(self,input_data,index):
        single_mask=torch.zeros(input_data.shape).bool()
        single_mask[index]=self.gene_chip
        return single_mask

    #use random mask to mask one person
    def random_single_mask(self,input_data,index):
        random_single_mask=torch.zeros(input_data.shape).bool()
        for i in index:
            random_single_mask[:,i]=self.random_mask(random_single_mask[:,i])
        return random_single_mask

    #calculate the maf
    def maf_cal(self,input_data):
        temp = np.sum(input_data, axis=1) / input_data.shape[1]
        self.maf_cal=torch.from_numpy(temp)

    # use maf mask to mask one person,index is size : [1,2,3],#maf is size [maf1,maf2]
    def maf_single_mask(self,input_data,index,maf):
        maf_mask = ((self.maf_cal <= maf[1]) & (self.maf_cal > maf[0]))
        maf_mask=(maf_mask & self.gene_chip)
        maf_single_mask = torch.zeros(input_data.shape).bool()
        maf_single_mask[index] = maf_mask
        return maf_single_mask