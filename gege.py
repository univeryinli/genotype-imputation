#!/usr/bin/python
# -*- coding: iso-8859-15 -*-

import os,json,torch,time,sys,copy,math,random
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
#from tensorboardX import SummaryWriter
import numpy as np
import pandas as pd
train_index=0
val_index=1


class Train_Data(Dataset):
    def __init__(self,input_data,target_data,window_matrix,masked):
        self.input_data=input_data
        self.target_data=target_data
        self.window_matrix=window_matrix
        self.masked=masked
        #self.target_masked=torch.ByteTensor([1 if i in self.masked else 0 for i in range(self.input_data.shape[0])])
        #print('mask:',self.target_masked.shape)

    def __len__(self):
        return len(self.window_matrix)

    def __getitem__(self, index):
        global train_index
        dic={}
        ix=self.window_matrix[index]
        dmodel = self.input_data.shape[1]
        encode_input=torch.from_numpy(self.input_data.loc[ix:ix+1000-1].values)
        positional_vector_encoder = torch.Tensor([[math.sin(
            math.pow((i + 1) / 1000, (j + 1) / dmodel)) if j % 2 == 0 else math.cos(
            math.pow((i + 1) / 1000, (j + 1) / dmodel)) for j in range(dmodel)] for i in range(ix, ix + 1000)])
        encode_input=encode_input.float()+positional_vector_encoder
        dic['encode_input']=encode_input

        dmodel=512
        decode_input=torch.from_numpy(self.target_data.iloc[ix:ix+1000,train_index].values)
        decode_target=decode_input
        positional_vector_decoder = torch.Tensor([[math.sin(
            math.pow((i + 1) / 1000, (j + 1) / dmodel)) if j % 2 == 0 else math.cos(
            math.pow((i + 1) / 1000, (j + 1) / dmodel)) for j in range(dmodel)] for i in range(ix, ix + 1000)])

        dic['positional_vector_decoder']=positional_vector_decoder
        target_masked=torch.rand(1000)>0.85
        decode_input=decode_input.masked_fill(target_masked,0)
        dic['decode_input'] = decode_input
        dic['target_masked']=target_masked
        dic['decode_target']=decode_target
        return dic


class Test_Data(Dataset):
    def __init__(self,input_data,target_data,window_matrix,masked):
        self.input_data=input_data
        self.target_data=target_data
        self.window_matrix=window_matrix
        #self.masked=masked
        #self.target_masked=torch.ByteTensor([1 if i in self.masked else 0 for i in range(self.input_data.shape[0])])

    def __len__(self):
        return len(self.window_matrix)

    def __getitem__(self, index):
        global val_index
        dic={}
        ix=self.window_matrix[index]
        dmodel = self.input_data.shape[1]
        encode_input=torch.from_numpy(self.input_data.loc[ix:ix+1000-1].values)
        positional_vector_encoder = torch.Tensor([[math.sin(
            math.pow((i + 1) / 1000, (j + 1) / dmodel)) if j % 2 == 0 else math.cos(
            math.pow((i + 1) / 1000, (j + 1) / dmodel)) for j in range(dmodel)] for i in range(ix, ix + 1000)])
        encode_input=encode_input.float()+positional_vector_encoder
        dic['encode_input']=encode_input

        dmodel=512
        decode_input=torch.from_numpy(self.target_data.iloc[ix:ix+1000,val_index].values)
        decode_target=decode_input
        positional_vector_decoder = torch.Tensor([[math.sin(
            math.pow((i + 1) / 1000, (j + 1) / dmodel)) if j % 2 == 0 else math.cos(
            math.pow((i + 1) / 1000, (j + 1) / dmodel)) for j in range(dmodel)] for i in range(ix, ix + 1000)])

        dic['positional_vector_decoder']=positional_vector_decoder
        target_masked=torch.rand(1000)>0.85
        decode_input=decode_input.masked_fill(target_masked,0)
        dic['decode_input'] = decode_input
        dic['target_masked']=target_masked
        dic['decode_target']=decode_target

        return dic


class Gene(nn.Module):
    def __init__(self):
        super(Gene,self).__init__()
        self.batch_n=nn.BatchNorm1d(5003)
        self.linear512_128=nn.Linear(512,128)
        self.linear128_5=nn.Linear(128,5)

        self.transformer=nn.Transformer(num_encoder_layers=1, num_decoder_layers=1)
        self.linear5003_2048=nn.Linear(5003,2048)
        self.linear2048_512=nn.Linear(2048,512)
        self.embeded=nn.Embedding(5,512)

    def forward(self, encode_input,decode_input,positional_vector,target_masked):
        encode_input=encode_input.transpose(2,1)
        encode_input=self.batch_n(encode_input)
        encode_input=encode_input.transpose(2,1)
        encode_input=self.linear5003_2048(encode_input)
        encode_input=F.relu(encode_input)
        encode_input=self.linear2048_512(encode_input)
        encode_input=F.relu(encode_input)
        decode_input_em=self.embeded(decode_input)
        decode_input_em=decode_input_em+positional_vector
        encode_input=encode_input.transpose(1,0)
        decode_input_em=decode_input_em.transpose(1,0)
        output=self.transformer(encode_input,decode_input_em,tgt_key_padding_mask=target_masked)
        output=F.relu(output)
        output=output.transpose(1,0)
        output=self.linear512_128(output)
        output=F.relu(output)
        output=self.linear128_5(output)
        return output



def train_model(models, dataloads, scheduler,criterion=None,history=None,num_epochs=100,board=None,
                train_steps=None, val_steps=None, model_save='val_loss',use_cuda=False,model_save_path='./',start_epoch=0):
    # train step 'None' is iter all the datasets ,if num ,will iter num step.
    since = time.time()

    best_model_wts = 0
    best_acc = 10.0
    best_loss = 10.0

    # print the para num in the network
    for model in models:
        print('#### the model para num is : ' , sum(p.numel() for p in model.parameters()))

    if start_epoch!=0:
        load_state=torch.load(os.path.join(model_save_path ,'1.best_model_wts'))
        start_epoch=load_state['epoch']
        state=load_state['state']
        for i,model in enumerate(models):
            model.load_state_dict(state[i])

    if torch.cuda.device_count() > 1 and use_cuda:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        models=[nn.DataParallel(model) for model in models]
        models=[model.cuda() for model in models]

        #load=torch.load('50k.1.best_model_wts')
        #models[0].load_state_dict(load[0])
        print('model is on cuda by gpus!')
    elif torch.cuda.device_count() == 1 and use_cuda:
        models = [model.cuda() for model in models]
        print('model is on cuda!')

    for epoch in range(start_epoch,num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        train_epoch_loss = 0.0
        train_epoch_acc = 0.0
        val_epoch_loss = 0.0
        val_epoch_acc = 0.0

        global train_index
        global val_index
        if epoch%4==0:
            train_index=random.randint(0, 2)
            val_index=random.randint(3,4)
        print('train_index:'+str(train_index)+','+'val_index'+str(val_index))

        # Ã¿Ò»žöµüŽú¶ŒÓÐÑµÁ·ºÍÑéÖ€œ×¶Î
        for phase in ['train', 'val']:
            steps = 0
            if phase == 'train':
                if train_steps==None:
                    steps=0
                else:
                    steps = train_steps
                # ÉèÖÃ training
                for model in models:
                    model.train(True)

            elif phase=='val':
                if val_steps==None:
                    steps=0
                else:
                    steps = val_steps
                # ÉèÖÃ evaluate
                for model in models:
                    model.train(False)

            epoch_loss = 0.0
            loss=torch.Tensor([0])
            epoch_acc = 0.0
            acc=torch.Tensor([0])
            stop_setps=0

            # ±éÀúÊýŸÝ
            for i, dataset in enumerate(dataloads[phase]):
                # »ñÈ¡ÊäÈë,Èç¹ûÊÇÓÐ¶àžöÊäÈëµÄ»°£¬¿ÉÒÔÓÃ×ÖµäÀŽÃèÊö¶àžöÊäÈë£¬È»ºóÍš¹ýkeyÀŽ»ñÈ¡Êä
                encode_input = dataset['encode_input']
                decode_input=dataset['decode_input']
                target_masked=dataset['target_masked']
                positional_vector_decoder=dataset['positional_vector_decoder']
                decode_target=dataset['decode_target']

                if board and epoch==start_epoch and phase=='train' and i==0:
                    if torch.cuda.device_count() > 1 and use_cuda:
                        print("Let's use", torch.cuda.device_count(), "GPUs!")
                        models = [model.module for model in models]
                        models = [model.cpu() for model in models]
                        board.add_graph(models[0], (encode_input,decode_input,positional_vector_decoder,target_masked))
                        models = [nn.DataParallel(model) for model in models]
                        models = [model.cuda() for model in models]


                # ÓÃcudaœøÐÐ°ü×°
                if use_cuda:
                    encode_input = encode_input.cuda()
                    decode_input = decode_input.cuda()
                    target_masked = target_masked.cuda()
                    positional_vector_decoder=positional_vector_decoder.cuda()
                    decode_target=decode_target.cuda()

                # ÉèÖÃ²ÎÊý
                scheduler.optimizer.zero_grad()

                # ÕýÏò
                output_v = models[0](encode_input,decode_input,positional_vector_decoder,target_masked)

                if phase=='train':
                    output=output_v
                    output = F.log_softmax(output, dim=2).view(-1,5)
                    output = F.nll_loss(output, decode_target.view(-1), reduction='none')
                    output = output.masked_select(target_masked.view(-1))
                    loss = output.sum()/output_v.shape[0]
                    acc=loss
                elif phase=='val':
                    output = output_v
                    output = F.log_softmax(output, dim=2).view(-1,5)
                    output = F.nll_loss(output, decode_target.view(-1), reduction='none')
                    output = output.masked_select(target_masked.view(-1))
                    loss = output.sum()/output_v.shape[0]
                    acc = loss
                # loss_tensor = criterion['loss'](decode_outputs, labels)
                # loss_ones=loss_tensor.new_ones(loss_tensor.size())
                # loss=((loss_ones-loss_tensor).sum())/batch_sizeaa
                # non_zeros=loss_tensor.nonzero().size()[0]
                # loss=(non_zeros-loss_tensor.sum())/non_zeros
                # acc = (criterion['acc'](decode_outputs, labels))
                #acc = loss

                # Èç¹ûÊÇÑµÁ·œ×¶ÎÏòºóŽ«µÝºÍÓÅ»¯,»á¶Ô¶àžöÄ£ÐÍµÄ²ÎÊýœøÐÐstep
                if phase == 'train':
                    loss.backward()
                    scheduler.optimizer.step()

                # Í³ŒÆËùÓÐloss
                epoch_loss += loss.data.item()
                epoch_acc += acc.data.item()

                if sys.version_info.major == 2:
                    sys.stdout.write('{} Loss: {:.4f} Acc: {:.4f}  Step:{:3d}/{:3d} \r'.format(
                        phase, loss.item(), acc.item(), i + 1, steps))
                    sys.stdout.flush()
                elif sys.version_info.major == 3:
                    print('{} Loss: {:.4f} Acc: {:.4f}  Step:{:3d}/{:3d}'.format(
                        phase, loss.item(), acc.item(), i + 1, steps), end='\n')

                if phase == 'train' and i == steps - 1:
                    break
                elif phase == 'val' and i == steps - 1:
                    break
                stop_setps=i
            print('\n')

            if phase == 'train':
                train_epoch_loss = epoch_loss / stop_setps
                train_epoch_acc = epoch_acc / stop_setps

            # epoch_acc=None
            elif phase == 'val':
                val_epoch_loss = epoch_loss / stop_setps
                val_epoch_acc = epoch_acc / stop_setps

            # Éî¿œ±ŽÄ£ÐÍ£¬µ±Ä£ÐÍ±íÏÖÐ§¹û±äºÃÊ±£¬ÐòÁÐ»¯Ä£ÐÍ
            if model_save=='train_loss':
                if train_epoch_loss < best_loss:
                    best_loss = train_epoch_loss
                    if torch.cuda.device_count()>1 and use_cuda:
                        best_model_wts = [copy.deepcopy(model.module).cpu().state_dict() for model in models]
                    else:
                        best_model_wts = [copy.deepcopy(model).cpu().state_dict() for model in models]
                else:
                    print('train loss model has not improving!')
            elif model_save=='val_loss':
                if val_epoch_loss < best_loss:
                    best_loss = val_epoch_loss
                    if torch.cuda.device_count() > 1 and use_cuda:
                        best_model_wts = [copy.deepcopy(model.module).cpu().state_dict() for model in models]
                    else:
                        best_model_wts = [copy.deepcopy(model).cpu().state_dict() for model in models]
                else:
                    print('val loss model has not improving!')

        scheduler.step()
        # ±£Žæ×îŒÑÄ£ÐÍµÄÈšÖØ£¬Èç¹ûÃ»ÓÐ×îŒÑÄ£ÐÍ£¬ÄÇÃŽ±£Žæ×îŒÑÄ£
        torch.save({'epoch':epoch,'state':best_model_wts,'comments':'random but 3train and 2val ,para cal,cover the masked position with0!'}, os.path.join(model_save_path ,'3.best_model_wts'))

        ## ±£ŽæÄ£ÐÍÈšÖØ£¬±ÈœÏÈšÖØËæÊ±Œä±ä»¯µÄ
        if board:
            if torch.cuda.device_count() > 1 and use_cuda:
                for name, para in models[0].module.named_parameters():
                    try:
                        board.add_histogram(name, para.clone().cpu().data.numpy(), epoch)
                    except:
                        board.add_histogram(name,np.zeros(para.shape),epoch)
            else:
                for name, para in models[0].named_parameters():
                    try:
                        board.add_histogram(name, para.clone().cpu().data.numpy(), epoch)
                    except:
                        board.add_histogram(name,np.zeros(para.shape),epoch)

        print('train: Loss: {:.4f} Acc: {:.4f} ;  val: Loss: {:.4f} Acc: {:.4f}'.format(
            train_epoch_loss, train_epoch_acc, val_epoch_loss, val_epoch_acc))
        print('\n')
        if board:
            try:
                board.add_scalars('train_val', {'train_loss': train_epoch_loss, 'train_acc': train_epoch_acc,'val_loss': val_epoch_loss, 'val_acc': val_epoch_acc}, epoch)
            except:
                continue

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:4f}'.format(best_loss))
    print('Best val Acc: {:4f}'.format(best_acc))
    if board:
        board.close()


def main():

    writer = SummaryWriter(comment='random but 3train and 2val ,para cal,cover the masked position with0!')
    input_data=pd.read_pickle('processed/chr9.phase3.impute.hap.100000.haplotype.train.pickle')
    targe_data=pd.read_pickle('processed/chr9.phase3.impute.hap.100000.haplotype.test.pickle')

    masked=np.int16(np.loadtxt('processed/masked.txt'))

    window_size=1000
    window_matrix=[i for i in range(0,input_data.shape[0]-window_size,window_size//16)]
    print('sample is:'+str(len(window_matrix)))
    train_dataset = Train_Data(input_data,targe_data,window_matrix,masked)
    test_dataset = Test_Data(input_data,targe_data,window_matrix,masked)
    data_loader = {'train':DataLoader(train_dataset,batch_size=16,shuffle=True,num_workers=64),'val':DataLoader(test_dataset,batch_size=16,shuffle=True,num_workers=64)}

    models=[Gene()]

    optim1=optim.SGD(models[0].parameters(),lr=0.001,momentum=0.9)
    lambda1=lambda epoch:epoch**0.95
    lr_sch=optim.lr_scheduler.LambdaLR(optim1,lr_lambda=lambda1)
    use_cuda=torch.cuda.is_available()
    print('cuda flag:  '+str(use_cuda))
    train_model(models,data_loader,lr_sch,board=writer,use_cuda=use_cuda)


if __name__ == '__main__':
    main()

