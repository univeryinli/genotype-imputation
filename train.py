import os,json,torch,time,sys,copy,math,random
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
from utils import r2_score,pearson



def train_model(model, dataloads, scheduler,criterion=None,num_epochs=14,board=None,
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

    step_wide=dataloads['train'].dataset.step
    print('step_wide:  ',step_wide)

    if board:
        board.add_graph(model, (dataloads['train'].dataset[0]['encode_input'].float().unsqueeze(0)))

    if start_epoch!=0:
        load_state=torch.load(os.path.join(model_save_path ,'6.best_model_wts'))
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

            # ????
            for i, dataset in enumerate(dataloads[phase]):
                encode_input=dataset['encode_input'].float()
                encode_mask=dataset['encode_mask']
                encode_target=dataset['encode_target'].float()
                encode_mask_target=encode_target.masked_select(encode_mask).float()

                if use_cuda:
                    encode_input=encode_input.cuda()
                    encode_mask=encode_mask.cuda()
                    encode_target=encode_target.cuda()
                    encode_mask_target=encode_mask_target.cuda()


                #
                scheduler.optimizer.zero_grad()

                #
                output_v = model(encode_input)
                loss2=0.1*F.binary_cross_entropy(output_v,encode_target)
                loss1=0.9*F.binary_cross_entropy(output_v.masked_select(encode_mask),encode_mask_target)
                loss = loss1+loss2

                acc = r2_score(output_v.masked_select(encode_mask),encode_mask_target)
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
        torch.save({'epoch':epoch,'state':best_model_wts,'comments':'1000G phase3_div'}, os.path.join(model_save_path ,'12.best_model_wts'))

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


def main():
    from preprocess import Mask,Train_data_div
    from unet import TrainData,TestData,Gene
    writer = SummaryWriter(comment='phase3_model')
    path = 'processed_phase3'
    gene_chip = np.load(os.path.join(path,'mask.npy'))
    input_data=np.load(os.path.join(path,'chr9.phase3.impute.hap.npy'))

    mask = Mask(gene_chip)
    mask.maf_cal(input_data)
    mask.missing_rate=0.99

    div_rate=0.7
    window_size = 1000
    train_data_div=Train_data_div(input_data,div_rate,path)
    train_data_div.training_index_save()

    train_data=input_data[:,train_data_div.train_index]
    val_data=input_data[:,train_data_div.val_index]

    train_dataset = TrainData(train_data,window_size,mask)
    val_dataset = TestData(val_data,window_size,mask)
    data_loader = {'train':DataLoader(train_dataset,batch_size=6,shuffle=True,drop_last=True,num_workers=16),'val':DataLoader(val_dataset,batch_size=6,shuffle=True,num_workers=16)}

    model=Gene(1,1)
    lr=0.01
    optim1=optim.SGD(model.parameters(),lr=lr,momentum=0.9)
    lr_sch=optim.lr_scheduler.LambdaLR(optim1,lr_lambda=lambda epoch:epoch*0.95)
    use_cuda=torch.cuda.is_available()
    print('cuda flag:  '+str(use_cuda))
    torch.cuda.empty_cache()
    train_model(model,data_loader,lr_sch,board=writer,use_cuda=use_cuda,lr=lr)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5'
    main()