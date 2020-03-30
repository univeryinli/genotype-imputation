import os,sys,torch
import numpy as np
from preprocess import Mask, Train_data_div
from unet import PreData, Gene


def predict_from_single_set(val_dataset):
    gene = Gene(1, 1)
    model=torch.load('runs/5.best_model_wts')
    model=model['state']
    gene.load_state_dict(model)
    gene=gene.cuda()

    import time
    start=time.time()
    result_sum=[]
    for i in range(len(val_dataset)):
        cal=gene(val_dataset[i].float().cuda(0))[0][0].detach().cpu()
        if i==0:
            result=cal[0:int(window_size*0.8)]
        elif i==(len(val_dataset)-1):
            result=cal[int(0.2*window_size):]
        else:
            result=cal[int(0.2*window_size):int(window_size*0.8)]
        result_sum.append(result)
        print(i, end='\r')
    result=torch.cat(result_sum)
    stop=time.time()
    time_use=stop-start
    print(time_use)
    return result


def correct_rate(pre,groud_s):
    pre_s=torch.round(pre).type(torch.int8)
    statis = pre_s == groud_s
    return statis.sum().item() / statis.size()[0]


def r2(pre,groud_s):
    tot = groud_s.float().var()
    res = ((groud_s.float() - pre.float()).abs() ** 2).mean()
    return 1 - (res / tot)


def predict_from_div_set():
    div_rate = 0.7
    #train_data_div = Train_data_div(input_data, div_rate, path)
    #train_data_div.training_index_load()


if __name__ == '__main__':
    path = 'processed_phase1'
    gene_chip = np.load(os.path.join(path,'mask.npy'))
    input_data = np.load(os.path.join(path, 'ALL.chr9.phase1.impute.hap.npy'))
    window_size = 1000
    mask = Mask(gene_chip)
    mask.maf_cal(input_data)
    mask.missing_rate=0.3
    val_dataset = PreData(input_data, window_size, mask)

    result=predict_from_single_set(val_dataset)
    torch.save(result, os.path.join(path, 'predict_result'))
    print('predict size is: ',result.shape)
    pre = result.masked_select(val_dataset.encode_mask)
    groud_s = torch.from_numpy(input_data).masked_select(val_dataset.encode_mask)
    print('correct rate is:  ',correct_rate(pre,groud_s) )
    print('r2 score is:  ', r2(pre,groud_s))