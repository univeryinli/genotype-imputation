# coding=GBK
import os, json, torch, time, sys, copy, math, random
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from utils import r2_score, pearson, auc


# 数据集文件，主要给模型喂数据
class DataSet(Dataset):
    def __init__(self, input_data, mask):
        self.mask = mask
        self.encode_mask = mask.gene_chip
        self.input_data = input_data

    def __len__(self):
        return self.input_data.shape[1]

    def __getitem__(self, people_index):
        dic = {}
        encode_target = self.input_data[:, people_index]
        encode_mask = self.encode_mask
        # encode_mask=self.mask.maf_random_mask(self.mask.maf_cal)
        encode_input = encode_target.masked_fill(encode_mask, 0)
        # encode_input = encode_target

        dic['encode_input'] = encode_input.unsqueeze(0)
        dic['encode_target'] = encode_target.unsqueeze(0)
        dic['encode_mask'] = encode_mask.unsqueeze(0)
        return dic


# 主要给预测模型喂数据，与训练方式喂数据的方式不同
class PreDataSet(Dataset):
    def __init__(self, input_data, mask):
        self.mask = mask
        self.encode_mask = mask.gene_chip
        self.input_data = input_data

    def __len__(self):
        return self.input_data.shape[1]

    def __getitem__(self, people_index):
        dic = {}
        encode_target = self.input_data[:, people_index]
        encode_mask = self.encode_mask
        encode_input = encode_target.masked_fill(encode_mask, 0)

        dic['encode_input'] = encode_input.unsqueeze(0)
        dic['encode_target'] = encode_target.unsqueeze(0)
        dic['encode_mask'] = encode_mask
        return dic


# 模型训练，这个是基于pytorch的通用神经网络训练模型
def train_model(model, dataloads, scheduler, num_epochs=25, board=None, cudaid=4,
                train_steps=None, val_steps=None, model_save='val_loss', use_cuda=False,
                model_save_path='./processed_phase3/model', start_epoch=0, lr=None, model_index=0):
    # train step 'None' is iter all the datasets ,if num ,will iter num step.
    # 训练时间起始标记
    since = time.time()

    # 指标参数初始化
    best_model_wts = 0
    best_acc = 0
    best_train_loss = 100
    best_val_loss = 100
    print('dataset-size:', len(dataloads['train']), len(dataloads['val']))

    # print the para num in the network 打印模型参数
    parasum = sum(p.numel() for p in model.parameters())
    print('#### the model para num is: ', parasum)

    if start_epoch != 0:
        load_state = torch.load(os.path.join(model_save_path, '5.best_model_wts'))
        start_epoch = load_state['epoch']
        state = load_state['state']
        model.load_state_dict(state)

    # 多GPU适配
    if torch.cuda.device_count() > 1 and use_cuda:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # choose which gpu runs手动模式，指定运行的GPU标识号
        if cudaid == -1:
            model = nn.DataParallel(model).cuda()
        else:
            model = model.cuda(cudaid)
        print('model is on cuda by gpus!')
    elif torch.cuda.device_count() == 1 and use_cuda:
        model = model.cuda()
        print('model is on cuda!')

    # 训练开始，按照轮数迭代
    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        train_epoch_loss = 0.0
        train_epoch_acc = 0.0
        val_epoch_loss = 0.0
        val_epoch_acc = 0.0

        # 迭代训练集和验证集
        for phase in ['train', 'val']:
            print(phase + ' steps is running!')
            if phase == 'train':
                if train_steps == None:
                    steps = 0
                else:
                    steps = train_steps
                # 模型训练按钮开启
                model = model.requires_grad_(True)

            elif phase == 'val':
                if val_steps == None:
                    steps = 0
                else:
                    steps = val_steps
                # 模型验证按钮开启
                model = model.requires_grad_(False)

            epoch_loss = 0.0
            epoch_acc = 0.0
            stop_setps = 0

            # 对训练集进行迭代训练
            for i, dataset in enumerate(dataloads[phase]):
                encode_input = dataset['encode_input'].float()
                # print(encode_input.shape)
                encode_target = dataset['encode_target'].float()
                # print(encode_target.shape)
                encode_mask = dataset['encode_mask']
                # print(encode_mask.shape)

                if torch.cuda.device_count() > 1 and use_cuda:
                    # choose which gpu runs手动模式，指定运行的GPU标识号
                    if cudaid == -1:
                        encode_input = encode_input.cuda()
                        encode_target = encode_target.cuda()
                        encode_mask = encode_mask.cuda()
                    else:
                        encode_input = encode_input.cuda(cudaid)
                        encode_target = encode_target.cuda(cudaid)
                        encode_mask = encode_mask.cuda(cudaid)
                # 参数优化初始化
                scheduler.optimizer.zero_grad()
                # 模型计算开始
                output_v = model(encode_input)
                # 损失计算
                # loss2=0.1*F.binary_cross_entropy(output_v.masked_select(~encode_mask),encode_target.masked_select(~encode_mask))
                # loss2 = 0.1 * F.mse_loss(output_v.masked_select(~encode_mask),encode_target.masked_select(~encode_mask))
                # loss1=0.9*F.binary_cross_entropy(output_v.masked_select(encode_mask),encode_target.masked_select(encode_mask))
                # loss1 = F.mse_loss(output_v.masked_select(encode_mask),encode_target.masked_select(encode_mask))
                loss = F.mse_loss(output_v, encode_target)
                # loss=F.binary_cross_entropy(output_v.squeeze(2),encode_target)
                # print(output_v.shape)
                # print(output_v.masked_select(~encode_mask).shape)
                # print(encode_target.shape)
                # print(encode_target.masked_select(~encode_mask).shape)
                # 评价指标计算
                acc = r2_score(output_v, encode_target)
                # 保存此时模型，model_index是模型的id号码，如果对多个窗口滑窗，那么不同窗保存不同模型参数
                torch.save(output_v, os.path.join('./processed_phase3/result', str(model_index)))

                # 迭代模型训练好的参数
                if phase == 'train':
                    loss.backward()
                    scheduler.optimizer.step()

                epoch_loss += loss.item()
                epoch_acc += acc.item()

                # 最终打印输出损失
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
                train_epoch_loss = epoch_loss / (stop_setps + 1)
                train_epoch_acc = epoch_acc / (stop_setps + 1)

            # epoch_acc=None
            elif phase == 'val':
                val_epoch_loss = epoch_loss / (stop_setps + 1)
                val_epoch_acc = epoch_acc / (stop_setps + 1)

        # 选择最佳模型参数，model_save参数控制选取的评价指标
        if model_save == 'train_loss':
            if train_epoch_loss < best_train_loss:
                best_train_loss = train_epoch_loss
                best_epoch = epoch
                if torch.cuda.device_count() > 1 and use_cuda and cudaid == -1:
                    best_model_wts = copy.deepcopy(model.module).cpu().state_dict()
                else:
                    best_model_wts = copy.deepcopy(model).cpu().state_dict()
            else:
                print('train loss model has not improving!')
        elif model_save == 'val_loss':
            if val_epoch_loss < best_val_loss:
                best_val_loss = val_epoch_loss
                best_acc = val_epoch_acc

                if torch.cuda.device_count() > 1 and use_cuda and cudaid == -1:
                    best_model_wts = copy.deepcopy(model.module).cpu().state_dict()
                else:
                    best_model_wts = copy.deepcopy(model).cpu().state_dict()
            else:
                print('val loss model has not improving!')

        scheduler.step()
        # 保存最佳模型
        torch.save({'epoch': epoch, 'state': best_model_wts, 'comments': '1000G phase3_div'},
                   os.path.join(model_save_path, str(model_index) + '.best_model_wts'))

        # 控制面板信息保存
        if board:
            if torch.cuda.device_count() > 1 and use_cuda and cudaid == -1:
                for name, para in model.module.named_parameters():
                    try:
                        board.add_histogram(name, para.clone().cpu().data.numpy(), epoch)
                    except:
                        board.add_histogram(name, np.zeros(para.shape), epoch)
            else:
                for name, para in model.named_parameters():
                    try:
                        board.add_histogram(name, para.clone().cpu().data.numpy(), epoch)
                    except:
                        board.add_histogram(name, np.zeros(para.shape), epoch)

        print('train: Loss: {:.4f} Acc: {:.4f} ;  val: Loss: {:.4f} Acc: {:.4f}'.format(
            train_epoch_loss, train_epoch_acc, val_epoch_loss, val_epoch_acc))
        print('\n')
        if board:
            try:
                board.add_scalars('all_', {'train_loss': train_epoch_loss, 'train_acc': train_epoch_acc,
                                           'val_loss': val_epoch_loss, 'val_acc': val_epoch_acc}, epoch)
            except:
                continue

    # 保存模型的超参数
    hpara = {'batch_size_train': dataloads['train'].batch_size, 'train_size': len(dataloads['train'].dataset), 'lr': lr,
             'num_para': parasum}
    mpara = {'best_train_loss': best_train_loss, 'best_val_loss': best_val_loss, 'best_val_acc': best_acc}
    print('add done!')
    board.add_hparams(hparam_dict=hpara, metric_dict=mpara)

    time_elapsed = time.time() - since
    # 打印最后汇总的训练结果
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best train Loss: {:4f}'.format(best_train_loss))
    print('Best val Loss: {:4f}'.format(best_val_loss))
    # 关闭最终的控制面板信息
    if board:
        board.close()
    # 返回最佳的指标值
    return best_acc


def train(i, rand):
    # 33000-36000
    from preprocess import Mask, Data_Div
    # from models.stack_transformer1d import Gene
    from models.cnn1d import Gene
    writer = SummaryWriter(comment='phase3_model')
    path = 'processed_phase3'
    gene_chip = torch.load(os.path.join(path, 'mask.high.torch'))[i:i + 1000]
    #gene_chip = torch.load(os.path.join(path, 'new_mask.torch'))[i:i + 1000]

    print('gene_chip', gene_chip.shape)
    input_data = torch.load(os.path.join(path, 'chr9.phase3.impute.high.hap.torch'))[i:i + 1000]
    #input_data = torch.load(os.path.join(path, 'chr9.phase3.impute.hap.torch'))[i:i + 1000]
    print('input_data', input_data.shape)

    mask = Mask(gene_chip)
    mask.maf_cal = mask.maf_cal(input_data)
    mask.missing_rate = 0.3
    print('missing_rate is: ', mask.missing_rate)

    data_div = Data_Div()
    col = [i for i in range(input_data.shape[1])]
    random.seed(1)
    data_div.study_panel = data_div.sampler(col, 0.05)
    data_div.reference_panel = list(set(col) - set(data_div.study_panel))

    train_data = input_data[:, data_div.reference_panel]
    val_data = input_data[:, data_div.study_panel]

    torch.random.manual_seed(rand)
    # mask.gene_chip=mask.maf_random_mask(mask.maf_cal)
    print('mask number is: ', mask.gene_chip.sum())

    train_dataset = DataSet(train_data, mask)
    print('train_samples is', len(train_dataset))
    val_dataset = DataSet(val_data, mask)
    print('val_sampler is', len(val_dataset))
    data_loader = {'train': DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=True, num_workers=48),
                   'val': DataLoader(val_dataset, batch_size=8, drop_last=True, shuffle=True, num_workers=48)}

    model = Gene()

    # 经过实验验证，当采用序列模型的时候，LR选取0.01比较合适，当采用cnn模型的时候，LR选取0.1或者0.2比较合适
    lr = 0.1
    optim1 = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    lr_sch = optim.lr_scheduler.LambdaLR(optim1, lr_lambda=lambda epoch: epoch * 0.95)
    use_cuda = torch.cuda.is_available()
    print('cuda flag:  ' + str(use_cuda))
    torch.cuda.empty_cache()
    train_model(model, data_loader, lr_sch, use_cuda=use_cuda, board=writer, lr=lr)


def predict(i, rand):
    # 33000-36000
    from models.stack_transformer1d import Gene
    path = 'processed_phase3'
    gene_chip = torch.load(os.path.join(path, 'mask.high.torch'))[i:i + 1000]
    print('gene_chip', gene_chip.shape)
    input_data = torch.load(os.path.join(path, 'chr9.phase3.impute.high.hap.torch'))[i:i + 1000]
    print('input_data', input_data.shape)

    mask = Mask(gene_chip)
    mask.maf_cal = mask.maf_cal(input_data)
    mask.missing_rate = 0.1

    data_div = Data_Div()
    col = [i for i in range(input_data.shape[1])]
    random.seed(1)
    data_div.study_panel = data_div.sampler(col, 0.05)
    data_div.reference_panel = list(set(col) - set(data_div.study_panel))

    val_data = input_data[:, data_div.study_panel]
    # torch.random.manual_seed(rand)
    # mask.gene_chip = mask.maf_random_mask(mask.maf_cal)
    print('mask number is: ', mask.gene_chip.sum())

    val_dataset = PreDataSet(val_data, mask)
    print('val_sampler is', len(val_dataset))

    gene = Gene()
    use_cuda = torch.cuda.is_available()
    print('cuda flag:  ' + str(use_cuda))
    torch.cuda.empty_cache()
    model_save_path = './processed_phase3/model'

    model = torch.load(os.path.join(model_save_path, str(0) + '.best_model_wts'))
    model = model['state']
    gene.load_state_dict(model)
    gene = gene.cuda()

    result = []
    target = []
    for i, dataset in enumerate(val_dataset):
        encode_input = dataset['encode_input'].float().unsqueeze(0)
        # print(encode_input.shape)
        # encode_target=dataset['encode_target'].float().unsqueeze(0)
        # print(encode_target.shape)
        # encode_mask=dataset['encode_mask']
        # print(encode_mask.shape)
        print(i)

        if use_cuda:
            encode_input = encode_input.cuda()
            # encode_target=encode_target.cuda()
            # encode_mask=encode_mask.cuda()

        output_v = gene(encode_input)
        result.append(output_v.detach())
        # target.append(encode_target)
    result = torch.cat(result, 0).squeeze(1).transpose(1, 0).cpu()
    # target=torch.cat(target,0).transpose(1,0).cpu()
    maf_list = [0.005, 0.05, 0.5]
    # maf_list = [0, 0.000001, 0.005, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

    for i in range(len(maf_list) - 1):
        maf_mask = ((mask.maf_cal <= maf_list[i + 1]) & (mask.maf_cal > maf_list[i]))
        print('maf_cal number is: ', maf_mask.sum())
        encode_mask = (mask.gene_chip & maf_mask).view(-1, 1)
        print('encode_mask is: ', sum(encode_mask))
        pre = result.masked_select(encode_mask)
        groud_s = val_data.masked_select(encode_mask)
        # print('correct rate is:  ',correct_rate(pre,groud_s))
        print('r2 score is:  ', r2_score(pre, groud_s))
        # print('pearson is :',pearson(pre,groud_s)[0]**2)
        print('auc score: ', auc(pre.detach().numpy(), groud_s.detach().numpy()))


def train_all():
    from preprocess import Mask, Data_Div
    # from transformer_encoder import Gene
    writer = SummaryWriter(comment='phase3_model')
    path = 'processed_phase3'
    gene_chip_test = torch.load(os.path.join(path, 'mask.high.torch'))
    print('gene_chip', gene_chip.shape)
    input_data_test = torch.load(os.path.join(path, 'chr9.phase3.impute.high.hap.torch'))
    print('input_data', input_data.shape)
    r = []

    for i in range(100):
        gene_chip = gene_chip_test[i:i + 1000]
        input_data = input_data_test[i:i + 1000]

        mask = Mask(gene_chip)
        mask.maf_cal = mask.maf_cal(input_data)
        mask.missing_rate = 0.87
        print('missing_rate is: ', mask.missing_rate)

        data_div = Data_Div()
        col = [i for i in range(input_data.shape[1])]
        random.seed(1)
        data_div.study_panel = data_div.sampler(col, 0.05)
        data_div.reference_panel = list(set(col) - set(data_div.study_panel))

        train_data = input_data[:, data_div.reference_panel]
        val_data = input_data[:, data_div.study_panel]

        torch.random.manual_seed(6)
        mask.gene_chip = mask.maf_random_mask(mask.maf_cal)
        print('mask number is: ', mask.gene_chip.sum())

        train_dataset = DataSet(train_data, mask)
        print('train_samples is', len(train_dataset))
        val_dataset = DataSet(val_data, mask)
        print('val_sampler is', len(val_dataset))
        data_loader = {'train': DataLoader(train_dataset, batch_size=4, shuffle=True, drop_last=True, num_workers=48),
                       'val': DataLoader(val_dataset, batch_size=4, drop_last=True, shuffle=True, num_workers=48)}

        model = Gene()
        lr = 0.1
        optim1 = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        lr_sch = optim.lr_scheduler.LambdaLR(optim1, lr_lambda=lambda epoch: epoch * 0.95)
        use_cuda = torch.cuda.is_available()
        print('cuda flag:  ' + str(use_cuda))
        torch.cuda.empty_cache()
        ac = train_model(model, data_loader, lr_sch, board=writer, use_cuda=False, lr=lr, model_index=i)
        r.append(ac)
    f = open('acc.txt', 'w')
    f.write(str(r))
    f.close()


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = '1,2,3,4,7'
    l = [6, 5, 4]
    for i in l:
        train(34000, rand=i)
        # predict(33000,rand=i)
