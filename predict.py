import os,sys,torch,random
import numpy as np
from utils import r2_score,correct_rate,pearson
from preprocess import Mask, Data_div
from unet import PreData, Gene


class PreData(Dataset):
    def __init__(self,input_data,data_div,window_size,mask):
        self.window_size=window_size
        self.step=int(window_size*0.6)
        self.window_matrix = [i for i in range(0, input_data.shape[0], self.step)]
        self.mask= mask

        self.index = list(set([i for i in range(input_data.shape[1])]) - set(data_div.train_index))
        self.people_index=[self.index.index(i) for i in data_div.val_index]
        self.input_data = input_data[:, self.index]
        # mask by the random mask
        # self.encode_mask=mask.random_single_mask(input_data,self.people_index)
        self.encode_mask = mask.chip_single_mask(input_data, self.index, data_div.val_index)
        # mask by the chip mask
        #self.encode_mask=mask.chip_single_mask(input_data,self.index,self.people_index)
        self.input_data = torch.from_numpy(self.input_data).masked_fill(self.encode_mask,-1)
        print('test ample is:' + str(len(self.window_matrix)))

    def __len__(self):
        return len(self.window_matrix)

    def __getitem__(self, index):
        ix = self.window_matrix[index]
        if ix>=(self.input_data.shape[0]-self.step):
            encode_input=self.input_data[ix:]
        else:
            encode_input = self.input_data[ix:ix + self.window_size]
        return encode_input.unsqueeze(0).unsqueeze(0)


def predict(val_dataset):
    gene = Gene(1, 1)
    model=torch.load('runs/2.best_model_wts')
    model=model['state']
    gene.load_state_dict(model)
    gene=gene.cuda(7)
    window_size=val_dataset.window_size
    import time
    start=time.time()
    result_sum=[]

    peopel_index=val_dataset.people_index
    print(val_dataset.people_index)
    for i in range(len(val_dataset)):
        cal=gene(val_dataset[i].float().cuda(7))[0][0].detach().cpu()
        if i==0:
            result=cal[:int(window_size*0.8),peopel_index]
        elif i==(len(val_dataset)-1):
            result=cal[int(0.2*window_size):,peopel_index]
        else:
            result=cal[int(0.2*window_size):int(window_size*0.8),peopel_index]
        result_sum.append(result)
        print(i, end='\r')
    result=torch.cat(result_sum)
    stop=time.time()
    time_use=stop-start
    print(time_use)
    return result


def main1():
    from preprocess import Mask, Data_div
    from unet import PreData, Gene
    path = 'processed_phase3'
    gene_chip = np.load(os.path.join(path, 'mask.npy'))
    input_data = np.load(os.path.join(path, 'chr9.phase3.impute.hap.npy'))

    mask = Mask(gene_chip)
    mask.missing_rate = 0.99
    print('missing_rate is:', mask.missing_rate)

    input_list = [i for i in range(input_data.shape[1])]
    div_rate = 0.995
    window_size = 1000
    random.seed(10)
    data_div = Data_div(path)
    data_div.reference_panel, data_div.study_panel = data_div.data_div(input_list, div_rate)

    mask.maf_cal(input_data[:, data_div.reference_panel])

    train_div = 0.8
    random.seed(9)
    data_div.train_index, data_div.val_index = data_div.data_div(data_div.study_panel, train_div)

    val_dataset = PreData(input_data, data_div, window_size, mask)
    result = predict(val_dataset)
    torch.save(result, os.path.join(path, 'predict_result'))
    print('predict size is: ', result.shape)
    # maf_list = [0.000001, 0.005, 0.05, 0.5]
    maf_list = [0, 0.000001, 0.005, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    for i in range(len(maf_list) - 1):
        maf_mask = ((mask.maf_cal <= maf_list[i + 1]) & (mask.maf_cal > maf_list[i]))
        encode_mask = (mask.gene_chip & maf_mask).view(-1, 1)
        pre = result.masked_select(encode_mask)
        pre = torch.round(pre)
        groud_s = torch.from_numpy(input_data[:, val_dataset.people_index]).masked_select(encode_mask)
        # print('correct rate is:  ',correct_rate(pre,groud_s))
        print('r2 score is:  ', r2_score(pre, groud_s))
        # print('pearson is :',pearson(pre,groud_s)[0]**2)


def main2():

    path = 'processed_phase3'
    gene_chip = np.load(os.path.join(path, 'mask.npy'))
    input_data = np.load(os.path.join(path, 'chr9.phase3.impute.hap.npy'))

    mask = Mask(gene_chip)
    mask.maf_cal(input_data)
    mask.missing_rate = 0.3

    window_size = 700
    data_div = Data_div(path)
    data_div.study_panel = [1000, 1001, 1200, 1201]

    data_div.train_index, data_div.val_index = data_div.study_panel[0:2], data_div.study_panel[2:4]

    val_dataset = PreData(input_data, data_div, window_size, mask)
    result = predict(val_dataset)
    torch.save(result, os.path.join(path, 'predict_result'))
    print('predict size is: ', result.shape)
    # maf_list = [0.000001, 0.005, 0.05, 0.5]
    maf_list = [0, 0.000001, 0.005, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    for i in range(len(maf_list) - 1):
        maf_mask = ((mask.maf_cal <= maf_list[i + 1]) & (mask.maf_cal > maf_list[i]))
        encode_mask = (mask.gene_chip & maf_mask).view(-1, 1)
        pre = result.masked_select(encode_mask)
        pre = torch.round(pre)
        groud_s = torch.from_numpy(input_data[:, val_dataset.people_index]).masked_select(encode_mask)
        # print('correct rate is:  ',correct_rate(pre,groud_s))
        print('r2 score is:  ', r2_score(pre, groud_s))
        # print('pearson is :',pearson(pre,groud_s)[0]**2)

if __name__ == '__main__':
   main2()