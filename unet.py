#!/usr/bin/python
# -*- coding: iso-8859-15 -*-

import os,json,torch,time,sys,copy,math,random
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import unet_parts
from unet_parts import *
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
        dic['encode_input'] = encode_input.unsqueeze(0)
        dic['encode_mask'] = encode_mask.unsqueeze(0)
        #encode_target = torch.zeros(5, encode_target.shape[0], encode_target.shape[1]).scatter(0,encode_target.unsqueeze(0),1)
        dic['encode_target']=encode_target.unsqueeze(0)
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
        dic['encode_input'] = encode_input.unsqueeze(0)
        dic['encode_mask'] = encode_mask.unsqueeze(0)
        #encode_target = torch.zeros(5, encode_target.shape[0], encode_target.shape[1]).scatter(0,encode_target.unsqueeze(0),1)
        dic['encode_target']=encode_target.unsqueeze(0)
        return dic


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
        return x


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

    if board:
        board.add_graph(model, (dataloads['train'].dataset[0]['encode_input'].float().unsqueeze(0)))

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


                # ????
                scheduler.optimizer.zero_grad()

                # ??
                output_v = model(encode_input)
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
        torch.save({'epoch':epoch,'state':best_model_wts,'comments':'u-net2'}, os.path.join(model_save_path ,'12.best_model_wts'))

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

    writer = SummaryWriter(comment='u-net for the gene')
    input_data=pd.read_pickle('processed/chr9.phase3.impute.hap.100000.haplotype.pickle')

    masked=np.int16(np.loadtxt('processed/masked.txt'))

    window_size=700
    length=input_data.shape[0]

    train_dataset = TrainData(input_data.loc[0:int(length*0.8)],window_size,masked)
    val_dataset = TestData(input_data.loc[int(length*0.8)+1:length],window_size,masked)
    data_loader = {'train':DataLoader(train_dataset,batch_size=8,shuffle=True,drop_last=True,num_workers=16),'val':DataLoader(val_dataset,batch_size=8,shuffle=True,num_workers=16)}

    model=Gene(1,1)
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

    window_size = 700
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
