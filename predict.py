import os,sys,torch,random
import numpy as np
from utils import r2_score,correct_rate,pearson


def predict(val_dataset):
    gene = Gene(1, 1)
    model=torch.load('runs/16.best_model_wts')
    model=model['state']
    gene.load_state_dict(model)
    gene=gene.cuda(7)

    import time
    start=time.time()
    result_sum=[]
    for i in range(len(val_dataset)):
        cal=gene(val_dataset[i].float().cuda(7))[0][0].detach().cpu()
        if i==0:
            result=cal[:int(window_size*0.8),val_dataset.people_index]
        elif i==(len(val_dataset)-1):
            result=cal[int(0.2*window_size):,val_dataset.people_index]
        else:
            result=cal[int(0.2*window_size):int(window_size*0.8),val_dataset.people_index]
        result_sum.append(result)
        print(i, end='\r')
    result=torch.cat(result_sum)
    stop=time.time()
    time_use=stop-start
    print(time_use)
    return result


if __name__ == '__main__':
    from preprocess import Mask, Data_div
    from unet import PreData, Gene
    path = 'processed_phase3'
    gene_chip = np.load(os.path.join(path,'mask.npy'))
    input_data = np.load(os.path.join(path, 'chr9.phase3.impute.hap.npy'))

    mask = Mask(gene_chip)
    mask.missing_rate=0.99
    print('missing_rate is:',mask.missing_rate)

    input_list = [i for i in range(input_data.shape[1])]
    div_rate = 0.9
    window_size = 1000
    random.seed(10)
    data_div = Data_div(path)
    data_div.reference_panel, data_div.study_panel = data_div.data_div(input_list, div_rate)

    mask.maf_cal(input_data[:,data_div.reference_panel])

    train_div = 0.8
    random.seed(9)
    data_div.train_index, data_div.val_index = data_div.data_div(data_div.study_panel, train_div)

    val_dataset = PreData(input_data,data_div,window_size,mask)
    result=predict(val_dataset)
    torch.save(result, os.path.join(path, 'predict_result'))
    print('predict size is: ',result.shape)
    maf_list = [0.000001, 0.005, 0.05, 0.5]
    #maf_list=[0,0.000001,0.005,0.05, 0.1 , 0.15, 0.2 , 0.25, 0.3 , 0.35, 0.4 , 0.45, 0.5 ]
    for i in range(len(maf_list)-1):
        maf_mask = ((mask.maf_cal <= maf_list[i+1]) & (mask.maf_cal > maf_list[i]))
        encode_mask=(mask.gene_chip & maf_mask).view(-1,1)
        pre = result.masked_select(encode_mask)
        pre=torch.round(pre)
        groud_s = torch.from_numpy(val_data[:,val_dataset.people_index]).masked_select(encode_mask)
        #print('correct rate is:  ',correct_rate(pre,groud_s))
        print('r2 score is:  ', r2_score(pre,groud_s))
        #print('pearson is :',pearson(pre,groud_s)[0]**2)