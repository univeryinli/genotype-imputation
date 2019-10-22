#!/usr/bin/python
# -*- coding: iso-8859-15 -*-

import os,json,torch,time,sys,copy,math,random
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import unet_parts
import numpy as np
import pandas as pd


class TrainData(Dataset):
    def __init__(self,input_data,window_size,masked):
        self.input_data=input_data
        self.window_size=window_size
        self.step=window_size//16
        self.window_matrix = [i for i in range(0, self.input_data.shape[0] - window_size, self.step)]
        self.missing_rate=0.15
        print('train ample is:' + str(len(self.window_matrix)))

    def __len__(self):
        return len(self.window_matrix)

    def __getitem__(self, index):

        dic = {}
        ix = self.window_matrix[index]
        encode_input = torch.from_numpy(self.input_data.loc[ix:ix + self.window_size-1].values)
        encode_target = encode_input
        encode_mask = torch.rand(encode_input.shape) > (1-self.missing_rate)
        encode_input = encode_input.masked_fill(encode_mask, 0)
        #encode_input = torch.zeros(5, encode_input.shape[0], encode_input.shape[1]).scatter(0, encode_input.unsqueeze(0), 1)
        dic['encode_input'] = encode_input
        dic['encode_mask'] = encode_mask
        #encode_target = torch.zeros(5, encode_target.shape[0], encode_target.shape[1]).scatter(0,encode_target.unsqueeze(0),1)
        dic['encode_target']=encode_target
        return dic


class TestData(Dataset):
    def __init__(self,input_data,window_size,masked):
        self.input_data=input_data
        self.window_size=window_size
        self.step=window_size//16
        self.window_matrix = [i for i in range(self.input_data.index[0], self.input_data.index[-1] - window_size, self.step)]
        self.missing_rate=0.15
        print('test ample is:' + str(len(self.window_matrix)))

    def __len__(self):
        return len(self.window_matrix)

    def __getitem__(self, index):

        dic = {}
        ix = self.window_matrix[index]
        encode_input = torch.from_numpy(self.input_data.loc[ix:ix + self.window_size-1].values)
        encode_target = encode_input
        encode_mask = torch.rand(encode_input.shape) > (1-self.missing_rate)
        encode_input = encode_input.masked_fill(encode_mask, 0)
        #encode_input = torch.zeros(5, encode_input.shape[0], encode_input.shape[1]).scatter(0, encode_input.unsqueeze(0), 1)
        dic['encode_input'] = encode_input
        dic['encode_mask'] = encode_mask
        #encode_target = torch.zeros(5, encode_target.shape[0], encode_target.shape[1]).scatter(0,encode_target.unsqueeze(0),1)
        dic['encode_target']=encode_target
        return dic


class Gene(nn.Module):
    def __init__(self, dmodel_in, dmodel_out,drop_out=0.1,nhead=8):
        super(Gene, self).__init__()
        self.dmodel_in=dmodel_in
        self.dmodel_out=dmodel_out
        #self.cnn1=nn.Conv2d(1,1,kernel_size=(1,3),padding=(0,1))
        #self.pool1=nn.MaxPool2d(kernel_size=(1,3),stride=(1,4))
        self.multi_head1=nn.MultiheadAttention(dmodel_in,nhead,dropout=drop_out)

        self.drop_out1_1=nn.Dropout(drop_out)
        self.norm1_1=nn.LayerNorm(dmodel_in)

        self.linear1=nn.Linear(dmodel_in,1024)

        self.multi_head2=nn.MultiheadAttention(1024,nhead,dropout=drop_out)
        self.drop_out2_1=nn.Dropout(drop_out)
        self.norm2_1=nn.LayerNorm(1024)

        self.linear2=nn.Linear(1024,256)

        self.multi_head3=nn.MultiheadAttention(256,nhead,dropout=drop_out)
        self.drop_out3_1=nn.Dropout(drop_out)
        self.norm3_1=nn.LayerNorm(256)

        self.linear3=nn.Linear(256,1024)

        self.multi_head4 = nn.MultiheadAttention(1024, nhead, dropout=drop_out)
        self.drop_out4_1 = nn.Dropout(drop_out)
        self.norm4_1 = nn.LayerNorm(1024)

        self.linear4 = nn.Linear(1024, dmodel_out)


    def forward(self, x,p1,p2,p3,p4):
        x=x.transpose(1,0)
        p1=p1.transpose(1,0)
        p2 = p2.transpose(1, 0)
        p3 = p3.transpose(1, 0)
        p4 = p4.transpose(1, 0)

        x=x+p1
        x1,_=self.multi_head1(x,x,x)
        x1=x+self.drop_out1_1(x1)
        x1=self.norm1_1(x1)

        x2=F.relu(self.linear1(x1),inplace=True)

        x2=x2+p2
        x3,_=self.multi_head2(x2,x2,x2)
        x3=x2+self.drop_out2_1(x3)
        x3=self.norm2_1(x3)

        x4=F.relu(self.linear2(x3),inplace=True)

        x4=x4+p3
        x5,_=self.multi_head3(x4,x4,x4)
        x5=x4+self.drop_out3_1(x5)
        x5=self.norm3_1(x5)

        x6=F.relu(self.linear3(x5),inplace=True)

        x6=x6+p4
        x7,_=self.multi_head4(x6,x6,x6)
        x7=x6+self.drop_out4_1(x7)
        x7=self.norm4_1(x7)

        x8=F.relu(self.linear4(x7),inplace=True)

        return x8.transpose(1,0)


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


def train_model(model, dataloads, scheduler,criterion=None,num_epochs=50,board=None,
                train_steps=None, val_steps=None, model_save='val_loss',use_cuda=False,model_save_path='./runs',start_epoch=0,lr=None):
    # train step 'None' is iter all the datasets ,if num ,will iter num step.
    since = time.time()

    best_model_wts = 0
    best_acc = 10.0
    best_train_loss = 100
    best_val_loss=100
    print('dataset-size:',len(dataloads['train']),len(dataloads['val']))
    # print the para num in the network
    parasum=sum(p.numel() for p in model.parameters())
    print('#### the model para num is: ' , parasum)
    missing_rate=dataloads['train'].dataset.missing_rate
    print('missing_rate:  ',missing_rate)
    step_wide=dataloads['train'].dataset.step
    print('step_wide:  ',step_wide)

    def position_v_g(dim,length):
        return torch.Tensor([[math.sin(
            math.pow((i + 1) / 1000, (j + 1) / dim)) if j % 2 == 0 else math.cos(
            math.pow((i + 1) / 1000, (j + 1) / dim)) for j in range(dim)] for i in range(length)])

    position_v1 = position_v_g(5008, 1000).expand(dataloads['train'].batch_size,1000,5008)
    position_v2 = position_v_g(1024, 1000).expand(dataloads['train'].batch_size,1000,1024)
    position_v3 = position_v_g(256, 1000).expand(dataloads['train'].batch_size,1000,256)
    position_v4 = position_v_g(1024, 1000).expand(dataloads['train'].batch_size,1000,1024)

    if board:
        board.add_graph(model, (dataloads['train'].dataset[0]['encode_input'].float().unsqueeze(0),position_v1[0:1],position_v2[0:1],position_v3[0:1],position_v4[0:1]))

    if start_epoch!=0:
        load_state=torch.load(os.path.join(model_save_path ,'1.best_model_wts'))
        start_epoch=load_state['epoch']
        state=load_state['state']
        model.load_state_dict(state[i])

    if torch.cuda.device_count() > 1 and use_cuda:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model=nn.DataParallel(model).cuda()

        #load=torch.load('50k.1.best_model_wts')
        #models[0].load_state_dict(load[0])
        print('model is on cuda by gpus!')
    elif torch.cuda.device_count() == 1 and use_cuda:
        model=model.cuda()
        print('model is on cuda!')

    for epoch in range(start_epoch,num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        train_epoch_loss = 0.0
        train_epoch_acc = 0.0
        val_epoch_loss = 0.0
        val_epoch_acc = 0.0

        # ??????????????
        for phase in ['train', 'val']:
            print(phase+' steps is running!')
            steps = 0
            if phase == 'train':
                if train_steps==None:
                    steps=0
                else:
                    steps = train_steps
                # ?? training
                model=model.requires_grad_(True)

            elif phase=='val':
                if val_steps==None:
                    steps=0
                else:
                    steps = val_steps
                # ?? evaluate
                model=model.requires_grad_(False)

            epoch_loss = 0.0
            epoch_acc = 0.0
            stop_setps=0

            # ????
            for i, dataset in enumerate(dataloads[phase]):
                # ????,????????????????????????????key????
                encode_input=dataset['encode_input'].float()
                encode_mask=dataset['encode_mask']
                encode_target=dataset['encode_target'].float()
                encode_mask_target=encode_target.masked_select(encode_mask).float()


                # ?cuda????n m
                if use_cuda:
                    encode_input=encode_input.cuda()
                    encode_mask=encode_mask.cuda()
                    encode_target=encode_target.cuda()
                    encode_mask_target=encode_mask_target.cuda()
                    position_v1=position_v1.cuda()
                    position_v2=position_v2.cuda()
                    position_v3=position_v3.cuda()
                    position_v4=position_v4.cuda()


                # ????
                scheduler.optimizer.zero_grad()

                # ??
                output_v = model(encode_input,position_v1,position_v2,position_v3,position_v4)
                loss2=0.1*F.mse_loss(output_v,encode_target)
                loss1=0.9*F.mse_loss(output_v.masked_select(encode_mask),encode_mask_target)
                loss = loss1+loss2

                # ??????????????,???????????step
                if phase == 'train':
                    loss.backward()
                    scheduler.optimizer.step()

                acc=loss
                # ????loss
                epoch_loss += loss.item()
                epoch_acc += acc.item()

                if sys.version_info.major == 2:
                    sys.stdout.write('{} Loss: {:.4f} Acc: {:.4f}  Step:{:3d}/{:3d} \r'.format(
                        phase, loss.item(), acc.item(), i + 1, steps))
                    sys.stdout.flush()
                elif sys.version_info.major == 3:
                    print('{} Loss: {:.4f} Acc: {:.4f}  Step:{:3d}/{:3d}'.format(
                        phase, loss.item(), acc.item(), i + 1, steps), end='\n')
                stop_setps = i
                if i == steps - 1:
                    break
            print('\n')

            if phase == 'train':
                train_epoch_loss = epoch_loss / (stop_setps+1)
                train_epoch_acc = epoch_acc / (stop_setps+1)

            # epoch_acc=None
            elif phase == 'val':
                val_epoch_loss = epoch_loss / (stop_setps+1)
                val_epoch_acc = epoch_acc / (stop_setps+1)

        # ??????????????????????
        if model_save=='train_loss':
            if train_epoch_loss < best_train_loss:
                best_train_loss = train_epoch_loss
                best_epoch=epoch
                if torch.cuda.device_count()>1 and use_cuda:
                    best_model_wts = copy.deepcopy(model.module).cpu().state_dict()
                else:
                    best_model_wts = copy.deepcopy(model).cpu().state_dict()
            else:
                print('train loss model has not improving!')
        elif model_save=='val_loss':
            if val_epoch_loss < best_val_loss:
                best_val_loss = val_epoch_loss
                best_epoch = epoch
                if torch.cuda.device_count() > 1 and use_cuda:
                    best_model_wts = copy.deepcopy(model.module).cpu().state_dict()
                else:
                    best_model_wts = copy.deepcopy(model).cpu().state_dict()
            else:
                print('val loss model has not improving!')

        scheduler.step()
        # ??????????????????????????
        torch.save({'epoch':epoch,'state':best_model_wts,'comments':'transformer-encoder'}, os.path.join(model_save_path ,'13.best_model_wts'))

        ## ?????????????????
        if board:
            if torch.cuda.device_count() > 1 and use_cuda:
                for name, para in model.module.named_parameters():
                    try:
                        board.add_histogram(name, para.clone().cpu().data.numpy(), epoch)
                    except:
                        board.add_histogram(name,np.zeros(para.shape),epoch)
            else:
                for name, para in model.named_parameters():
                    try:
                        board.add_histogram(name, para.clone().cpu().data.numpy(), epoch)
                    except:
                        board.add_histogram(name,np.zeros(para.shape),epoch)

        print('train: Loss: {:.4f} Acc: {:.4f} ;  val: Loss: {:.4f} Acc: {:.4f}'.format(
            train_epoch_loss, train_epoch_acc, val_epoch_loss, val_epoch_acc))
        print('\n')
        if board:
            try:
                board.add_scalars('all_', {'train_loss': train_epoch_loss, 'train_acc': train_epoch_acc,'val_loss': val_epoch_loss, 'val_acc': val_epoch_acc}, epoch)
            except:
                continue

    hpara = {'batch_size_train': dataloads['train'].batch_size,'window_size': dataloads['train'].dataset.window_size,
             'train_size': len(dataloads['train'].dataset),'lr':lr,'num_para':parasum,'missing_rate':missing_rate,'step_wide':step_wide}
    mpara={'best_train_loss': best_train_loss, 'best_val_loss': best_val_loss}
    print('add done!')
    board.add_hparams(hparam_dict=hpara,metric_dict=mpara)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best train Loss: {:4f}'.format(best_train_loss))
    print('Best val Loss: {:4f}'.format(best_val_loss))
    if board:
        board.close()


def main():

    writer = SummaryWriter(comment='transformer-encoder')
    input_data=pd.read_pickle('processed/chr9.phase3.impute.hap.100000.haplotype.pickle')

    masked=np.int16(np.loadtxt('processed/masked.txt'))

    window_size=1000
    length=input_data.shape[0]

    train_dataset = TrainData(input_data.loc[0:int(length*0.8)],window_size,masked)
    val_dataset = TestData(input_data.loc[int(length*0.8)+1:length],window_size,masked)
    data_loader = {'train':DataLoader(train_dataset,batch_size=16,shuffle=True,drop_last=True,num_workers=16),'val':DataLoader(val_dataset,batch_size=16,shuffle=True,drop_last=True,num_workers=16)}

    model=Gene(5008,5008)
    lr=0.01
    optim1=optim.SGD(model.parameters(),lr=lr,momentum=0.9)
    lr_sch=optim.lr_scheduler.LambdaLR(optim1,lr_lambda=lambda epoch:epoch*0.95)
    use_cuda=torch.cuda.is_available()
    print('cuda flag:  '+str(use_cuda))
    torch.cuda.empty_cache()
    train_model(model,data_loader,lr_sch,board=writer,use_cuda=use_cuda,lr=lr)


def predict():
    input_data = pd.read_pickle('processed/chr9.phase3.impute.hap.100000.haplotype.pickle')

    masked = np.int16(np.loadtxt('processed/masked.txt'))

    window_size = 1000
    length = input_data.shape[0]

    train_dataset = TrainData(input_data.loc[0:int(length * 0.8)], window_size, masked)
    val_dataset = TestData(input_data.loc[int(length * 0.8) + 1:length], window_size, masked)
    data_loader = {'train': DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=True, num_workers=16),
                   'val': DataLoader(val_dataset, batch_size=8, shuffle=True, num_workers=16)}

    model = Gene(1, 1)
    model.load
    for i,dataset in enumerate(data_loader):
        dataset=model


if __name__ == '__main__':
    main()
