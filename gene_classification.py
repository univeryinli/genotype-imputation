import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import os,json,torch,time,sys,copy
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
#from tensorboardX import SummaryWriter


class Train_Data(Dataset):
    def __init__(self,train_path):
        if train_path and not os.path.exists(train_path):
            print("Save path is not existed!")
            return
        self.train_path=train_path
        self.train_path_list=os.listdir(train_path)

    def __len__(self):
        return len(self.train_path_list)

    def __getitem__(self, index):
        item=os.path.join(self.train_path,self.train_path_list[index])
        dic=json.load(open(item,'r'))
        dic['input']=torch.LongTensor(dic['input'])
        return dic


class Test_Data(Dataset):
    def __init__(self,test_path):
        if test_path and not os.path.exists(test_path):
            print("Save path is not existed!")
            return
        self.test_path=test_path
        self.test_path_list=os.listdir(test_path)

    def __len__(self):
        return len(self.test_path_list)

    def __getitem__(self, index):
        item=os.path.join(self.test_path,self.test_path_list[index])
        dic=json.load(open(item,'r'))
        dic['input']=torch.LongTensor(dic['input'])
        return dic


class Gene(nn.Module):
    def __init__(self):
        super(Gene,self).__init__()
        self.embed=nn.Embedding(5,4)
        self.lstm1=nn.LSTM(4,32,1)
        self.lstm2=nn.LSTM(32,128,1)
        self.lstm3=nn.LSTM(128,256,1)
        self.lstm4=nn.LSTM(256,256,1)
        self.linear1=nn.Linear(256,128)
        self.linear2=nn.Linear(128,32)
        self.linear3=nn.Linear(32,5)


    def forward(self, input):
        embed=self.embed(input)
        embed=embed.transpose(0,1)
        ls1,_=self.lstm1(embed)
        ls2,_=self.lstm2(ls1)
        ls3,_=self.lstm3(ls2)
        _,(ls4,_)=self.lstm4(ls3)
        ln1=F.relu(self.linear1(ls4))
        ln2=F.relu(self.linear2(ln1))
        ln3=F.relu(self.linear3(ln2))
        softmax=F.log_softmax(ln3,dim=2)
        softmax=torch.sum(softmax,dim=0)
        return softmax


def train_model(models, dataloads, scheduler,criterion=None,history=None,num_epochs=100,board=None,
                train_steps=None, val_steps=64, val_model='loss',use_cuda=False,model_save_path='./',start_epoch=0):

    since = time.time()
    test_input=torch.randint(4,(2,256))
    board.add_graph(models[0], (test_input,))
    best_model_wts = 0
    best_acc = 10.0
    best_loss = 10.0

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
        load=torch.load('50k.1.best_model_wts')
        models[0].load_state_dict(load[0])
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

        # 每一个迭代都有训练和验证阶段
        for phase in ['train', 'val']:
            steps = 0
            if phase == 'train':
                # scheduler.step()
                if train_steps==None:
                    steps=0
                else:
                    steps = train_steps
                scheduler.step()
                # 设置 model 为训练 (training) 模式
                for model in models:
                    model.train(True)

            elif phase=='val':
                if val_steps==None:
                    steps=0
                else:
                    steps = val_steps
                # 设置 model 为评估 (evaluate) 模式
                for model in models:
                    model.train(False)

            epoch_loss = 0.0
            loss=torch.Tensor([0])
            epoch_acc = 0.0
            acc=torch.Tensor([0])
            stop_setps=0

            # 遍历数据
            for i, dataset in enumerate(dataloads[phase]):
                # 获取输入,如果是有多个输入的话，可以用字典来描述多个输入，然后通过key来获取输入
                input = dataset['input']
                lable=dataset['lable']

                # 用 cuda进行包装
                if use_cuda:
                    if phase == 'train':
                        input = input.cuda()
                        lable = lable.cuda()
                    elif phase == 'val':
                        input = input.cuda()
                        lable = lable.cuda()
                        # labels = Variable(targets.cuda())

                # 设置梯度参数为0
                scheduler.optimizer.zero_grad()

                # 正向传递
                output = models[0](input)
                if phase=='train':
                    loss = F.nll_loss(output,lable)
                    acc=loss
                elif phase=='val':
                    loss=F.nll_loss(output,lable)
                    acc = loss
                # loss_tensor = criterion['loss'](decode_outputs, labels)
                # loss_ones=loss_tensor.new_ones(loss_tensor.size())
                # loss=((loss_ones-loss_tensor).sum())/batch_sizeaa
                # non_zeros=loss_tensor.nonzero().size()[0]
                # loss=(non_zeros-loss_tensor.sum())/non_zeros
                # acc = (criterion['acc'](decode_outputs, labels))
                #acc = loss

                # 如果是训练阶段, 向后传递和优化,会对多个模型的参数进行step
                if phase == 'train':
                    loss.backward()
                    scheduler.optimizer.step()

                # 统计
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

            if train_epoch_loss < best_loss:
                best_loss = train_epoch_loss
                best_model_wts = [copy.deepcopy(model.module).cpu().state_dict() for model in models]
            else:
                print('model has not improving!')
            ''''# 深拷贝 model,两种模型正确率评价模式
            if val_model == 'acc' and phase == 'val':
                if val_epoch_acc < best_acc:
                    best_acc = val_epoch_acc
                    best_model_wts = (copy.deepcopy(encoder.state_dict()), copy.deepcopy(decoder.state_dict()))
                else:
                    print('model has not improving!')
            elif val_model == 'loss' and phase == 'val':
                if val_epoch_loss < best_loss:
                    best_loss = val_epoch_loss
                    best_model_wts = (copy.deepcopy(encoder.state_dict()), copy.deepcopy(decoder.state_dict()))
                else:
                    print('model has not improving!')'''

        # 保存最佳模型的权重
        torch.save({'epoch':epoch,'state':best_model_wts}, os.path.join(model_save_path ,'1.best_model_wts'))

        ## 保存模型权重，比较权重随时间变化的曲线
        if torch.cuda.device_count() > 1 and use_cuda:
            for name, para in models[0].module.named_parameters():
                board.add_histogram(name, para.clone().cpu().data.numpy(), epoch)
        else:
            for name, para in models[0].named_parameters():
                board.add_histogram(name, para.clone().cpu().data.numpy(), epoch)


        # # 保存loss相关数据
        # if history !=None:
        #     history.history['train_loss'].append(train_epoch_loss)
        #     history.history['train_acc'].append(train_epoch_acc)
        #     history.history['val_loss'].append(val_epoch_loss)
        #     history.history['val_acc'].append(val_epoch_acc)

        print('train: Loss: {:.4f} Acc: {:.4f} ;  val: Loss: {:.4f} Acc: {:.4f}'.format(
            train_epoch_loss, train_epoch_acc, val_epoch_loss, val_epoch_acc))
        print('\n')
        board.add_scalars('train_val', {'train_loss': train_epoch_loss, 'train_acc': train_epoch_acc,'val_loss': val_epoch_loss, 'val_acc': val_epoch_acc}, epoch)

    # # 保存loss到文件
    # if not history:
    #     history.save( model_save_path+ '50k.1.loss')

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:4f}'.format(best_loss))
    print('Best val Acc: {:4f}'.format(best_acc))
    board.close()


def main():
    writer = SummaryWriter()
    train_path = './gene_train'
    test_path = './gene_test'

    train_dataset = Train_Data(train_path)
    test_dataset = Test_Data(test_path)
    data_loader = {'train':DataLoader(train_dataset,batch_size=16,shuffle=True,num_workers=4),'val':DataLoader(test_dataset,batch_size=16,shuffle=True,num_workers=4)}

    models=[Gene()]

    optim1=optim.SGD(models[0].parameters(),lr=0.01,momentum=0.9)
    lambda1=lambda epoch:epoch**0.95
    lr_sch=optim.lr_scheduler.LambdaLR(optim1,lr_lambda=lambda1)
    use_cuda=torch.cuda.is_available()
    train_model(models,data_loader,lr_sch,board=writer,use_cuda=use_cuda)


if __name__ == '__main__':
    main()
