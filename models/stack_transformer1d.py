#coding=GBK
import os,json,torch,time,sys,copy,math,random
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import torch.optim as optim
import numpy as np


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout=0.5, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp((torch.arange(0, d_model, 2) *-(math.log(10000.0) / d_model)).float())
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]

        return self.dropout(x)


class Gene(nn.Module):
    def __init__(self):
        super(Gene,self).__init__()
        # self.emed = nn.Embedding(3, 3, padding_idx=0)
        self.linear1=nn.Linear(3,16)
        self.rnn1 = nn.Transformer(d_model=16, nhead=4, num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=32)
        self.linear2=nn.Linear(16,1)
        sz=self.generate_square_subsequent_mask(1000)
        self.po = PositionalEncoding(16, max_len=1000)
        self.register_buffer('sz', sz)

    def forward(self, x):
        x=x.transpose(2,1).long()
        x = torch.zeros((x.shape[0], x.shape[1],3),device=x.device).scatter_(2, x, 1)
        x=self.linear1(x)
        x=self.po(x)
        x=x.transpose(1,0)
        x=self.rnn1(x,x,tgt_mask=self.sz)
        x=x.transpose(1,0)
        x = self.linear2(x)
        x=F.relu(x,inplace=True)
        return x.transpose(2,1)

    def generate_square_subsequent_mask(self,size):
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask