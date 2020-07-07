#!/usr/bin/python
# -*- coding: iso-8859-15 -*-
import os,json,torch,time,sys,copy,math,random
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import torch.optim as optim
import numpy as np
import pandas as pd
import os,json,torch,time,sys,copy,math,random
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
from utils import r2_score,pearson

class TrainData(Dataset):
    def __init__(self,input_data,data_div,window_size,mask):
        self.data_div=data_div
        self.mask=mask
        self.window_size=window_size
        self.step=int(window_size*0.6)
        self.window_matrix = [i for i in range(0, input_data.shape[0] - window_size, self.step)]
        self.encode_mask = mask.gene_chip
        self.index=list(set([i for  i in range(input_data.shape[1])])-set(data_div.study_panel))
        self.encode_input1 = input_data[:,self.index]
        self.col=[i for i in range(len(self.index))]
        print('train ample is:' + str(len(self.window_matrix)))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, people_index):
        dic = {}
        ix = self.window_matrix[ix_index]
        encode_input = self.encode_input1[ix:ix + self.window_size,people_index].clone()
        encode_mask = self.encode_mask[ix:ix + self.window_size]
        encode_input = encode_input.masked_fill(encode_mask,2)
        encode_target=self.encode_input1[ix:ix + self.window_size,people_index]

        dic['encode_input'] = encode_input
        dic['encode_target']=encode_target
        dic['encode_mask']=encode_mask
        return dic


class TestData(Dataset):
    def __init__(self,input_data,data_div,window_size,mask):
        self.data_div=data_div
        self.mask=mask
        self.window_size=window_size
        self.step=int(window_size*0.6)
        self.window_matrix = [i for i in range(0, input_data.shape[0] - window_size, self.step)]
        self.encode_mask = mask.gene_chip
        self.index=data_div.study_panel
        self.encode_input1 = input_data[:,self.index]
        print('train ample is:' + str(len(self.window_matrix)))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, people_index):
        dic = {}
        ix = self.window_matrix[ix_index]
        encode_input = self.encode_input1[ix:ix + self.window_size,people_index].clone()
        encode_target=self.encode_input1[ix:ix + self.window_size,people_index]
        encode_mask = self.encode_mask[ix:ix + self.window_size]
        encode_input = encode_input.masked_fill(encode_mask, 2)
        dic['encode_input'] = encode_input
        dic['encode_target']=encode_target
        dic['encode_mask']=encode_mask
        return dic


class Gene(nn.Module):
    def __init__(self):
        super(Gene, self).__init__()
        self.conv = nn.Sequential(nn.Conv1d(1, 16, 5, stride=3, padding=0),
                                  nn.Conv1d(16, 32, 5, stride=3, padding=0),
                                  nn.Conv1d(32, 64, 5, stride=3, padding=0),
                                  nn.Conv1d(64, 128, 3, stride=2, padding=0),
                                  nn.Conv1d(128, 256, 3, stride=2, padding=0))
        self.pool = nn.MaxPool1d(8)
        self.linear = nn.Linear(256, 1000)

    def forward(self, x1):
        x1 = self.conv(x1)
        print(x1.shape)
        x1 = self.pool(x1)
        print(x1.shape)
        x1 = x1.squeeze(2)
        x1 = self.linear(x1)
        return x1


def train_model(model, dataloads, scheduler,num_epochs=70,board=None,
                train_steps=None, val_steps=1, model_save='val_loss',use_cuda=False,model_save_path='./runs',start_epoch=0,lr=None):
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

    step_wide=dataloads['train'].dataset.step
    print('step_wide:  ',step_wide)

    if start_epoch!=0:
        load_state=torch.load(os.path.join(model_save_path ,'4.best_model_wts'))
        start_epoch=load_state['epoch']
        state=load_state['state']
        model.load_state_dict(state)

    if torch.cuda.device_count() > 1 and use_cuda:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model=nn.DataParallel(model).cuda()
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

            #
            for i, dataset in enumerate(dataloads[phase]):
                encode_input=dataset['encode_input']
                #print(encode_input.shape)
                encode_target=dataset['encode_target'].float()
                #print(encode_target.shape)
                encode_mask=dataset['encode_mask']
                #print(encode_mask.shape)

                if use_cuda:
                    encode_input=encode_input.cuda()
                    encode_target=encode_target.cuda()
                    encode_mask=encode_mask.cuda()

                #
                scheduler.optimizer.zero_grad()

                #
                output_v = model(encode_input)
                #loss2=0.1*F.binary_cross_entropy(output_v.masked_select(~encode_mask),encode_target.masked_select(~encode_mask))
                #loss2 = 0.1 * F.mse_loss(output_v.masked_select(~encode_mask),encode_target.masked_select(~encode_mask))
                #loss1=0.9*F.binary_cross_entropy(output_v.masked_select(encode_mask),encode_target.masked_select(encode_mask))
                #loss1 = F.mse_loss(output_v.masked_select(encode_mask),encode_target.masked_select(encode_mask))
                #loss = F.mse_loss(output_v,encode_target)
                loss=F.binary_cross_entropy(output_v.squeeze(2),encode_target)

                acc = r2_score(output_v.squeeze(2).masked_select(encode_mask),encode_target.masked_select(encode_mask))
                # step
                if phase == 'train':
                    loss.backward()
                    scheduler.optimizer.step()

                epoch_loss += loss.item()
                epoch_acc += acc.item()

                if sys.version_info.major == 2:
                    sys.stdout.write('{} Loss: {:.4f} Acc: {:.4f}  Step:{:3d}/{:3d} \r'.format(
                        phase, loss.item(), acc.item(), i + 1, steps))
                    sys.stdout.flush()
                elif sys.version_info.major == 3:
                    print('{} Loss: {:.4f} Acc: {:.4f}  Step:{:3d}/{:3d}'.format(
                        phase, loss.item(), acc.item(), i + 1, steps), end='\r')
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

        # model save mode
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
        # save model
        torch.save({'epoch':epoch,'state':best_model_wts,'comments':'1000G phase3_div'}, os.path.join(model_save_path ,'6.best_model_wts'))

        # board save
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
             'train_size': len(dataloads['train'].dataset),'lr':lr,'num_para':parasum,'step_wide':step_wide}
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
        
        
def main2():
    from preprocess import Mask,Data_div
    #from transformer_encoder import Gene
    writer = SummaryWriter(comment='phase3_model')
    path = 'processed_phase3'
    gene_chip = np.load(os.path.join(path,'mask.npy'))
    input_data=np.load(os.path.join(path,'chr9.phase3.impute.hap.npy'))[:2000000,:]

    mask = Mask(gene_chip)
    mask.maf_cal(input_data)
    mask.missing_rate=0.3

    window_size = 256
    data_div=Data_div(path)
    col=[i for i in range(input_data.shape[1])]
    random.seed(10)
    data_div.study_panel=random.sample(col,4)
    data_div.train_index=random.sample(data_div.study_panel,2)
    data_div.val_index=list(set(data_div.study_panel)-set(data_div.train_index))

    input_data=torch.from_numpy(input_data)
    train_dataset=TrainData(input_data,data_div,window_size,mask)
    val_dataset = TestData(input_data,data_div,window_size,mask)
    data_loader = {'train':DataLoader(train_dataset,batch_size=8,shuffle=True,drop_last=True,num_workers=8),'val':DataLoader(val_dataset,batch_size=2,drop_last=True,shuffle=False,num_workers=8)}

    model=Gene()
    lr=0.01
    optim1=optim.SGD(model.parameters(),lr=lr,momentum=0.9)
    lr_sch=optim.lr_scheduler.LambdaLR(optim1,lr_lambda=lambda epoch:epoch*0.95)
    use_cuda=torch.cuda.is_available()
    print('cuda flag:  '+str(use_cuda))
    torch.cuda.empty_cache()
    train_model(model,data_loader,lr_sch,board=writer,use_cuda=False,lr=lr)
    

if __name__ == '__main__':
    #os.environ["CUDA_VISIBLE_DEVICES"] = '1,2,3,4,7'
    main2()