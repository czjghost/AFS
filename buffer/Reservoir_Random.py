# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 16:38:37 2022

@author: czjghost
"""
import torch
import os
import numpy as np
import torch.nn as nn
#import sys,os
#sys.path.append(os.path.dirname(__file__) + os.sep + '../')
#from parameter import parameter_setting #导入父级文件夹的parameter
#from setup_elements import setup_architecture
from ipdb import set_trace
from torchvision import transforms
from copy import deepcopy
import random
import time

input_size_match = {
    'cifar10': (3, 32, 32),
    'cifar100': (3, 32, 32),
    'core50': (3, 128, 128),
    'mini_imagenet': (3, 84, 84)
}
feature_size_match = {
    'cifar100': 160,
    'cifar10': 160,
    'core50': 2560,
    'mini_imagenet': 640,
}

def initial(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    
def single_euclidean_dist(x, y):#不开根号的欧式距离
#    d = ((x - y)**2).sum(0).clamp(min=1e-12).sqrt()
    d = ((x - y)**2).sum(0)
    return d

def euclidean_dist(x, y):#x与y两两计算距离的快捷方法
    #x: mean feature, y : test feature
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist = dist - 2 * (x @ y.t())
#    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def Random_Retrieve(cur_idx, batch_size, buffer_img, buffer_label):
    #random retrieve
    all_index = np.arange(cur_idx)
    #replace用来设置是否选取相同的样本
    select_batch_size = min(batch_size, cur_idx)
    select_index = torch.from_numpy(np.random.choice(all_index, select_batch_size, replace=False)).long().cuda()

    x = buffer_img[select_index]
    y = buffer_label[select_index]
    return x,y


class Reservoir_Random(object):
    
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.cur_idx = 0#当前的内存中已存储的样本集
        self.n_class_seen = 0
        self.cls_set = {}
        self.all_classes = params.num_classes
        
        self.buffer_size = params.mem_size
        self.input_size = input_size_match[params.data]
        self.feat_size = feature_size_match[params.data]
        
        #buff cache, 设置float可以加“, dtype=torch.float”
        self.buffer_img = torch.zeros((self.buffer_size,) + self.input_size).float().cuda()
        self.buffer_label = torch.zeros(self.buffer_size).long().cuda()

        #used for Reservoir Sampling counts
        self.n_sample_seen_so_far = 0 
        
        self.mean_feat = torch.zeros((0, self.feat_size)).float().cuda()
        self.mean_feat_label = torch.zeros(0).long().cuda()
        
    def is_empty(self):
        if self.cur_idx > 0:
            return False
        return True
    
    def update(self, x_train, y_train):#默认设置为蓄水池采样法
#        set_trace() #用来调试
        n = x_train.size(0)
        for i in range(n):#后续改为批量处理
            if self.cur_idx < self.buffer_size:
                self.buffer_img[self.cur_idx] = x_train[i]

                self.buffer_label[self.cur_idx] = y_train[i]
                self.cur_idx += 1
                
                if int(y_train[i]) in self.cls_set:# 对应key值的value加一
                    self.cls_set[int(y_train[i])] += 1
                else:# 第一次出现则创建新key
                    self.cls_set[int(y_train[i])] = 1
                    self.n_class_seen += 1
            
            else:
                r_idx = np.random.randint(0, self.n_sample_seen_so_far + i)

                if r_idx < self.buffer_size:
                    #要替换的 img的标签的key的value减一
                    self.cls_set[int(self.buffer_label[r_idx])] -= 1
                    
                    if self.cls_set[int(self.buffer_label[r_idx])] == 0:# 最后一个出现了
                        self.cls_set.pop(int(self.buffer_label[r_idx]))
                        self.n_class_seen -= 1
                    
                    self.buffer_img[r_idx] = x_train[i]
                    self.buffer_label[r_idx] = y_train[i]
                    
                    if int(y_train[i]) in self.cls_set:# 对应key值的value加一
                        self.cls_set[int(y_train[i])] += 1
                    else:# 第一次出现则创建新key
                        self.cls_set[int(y_train[i])] = 1
                        self.n_class_seen += 1
                        
            #update total number of sample have seen so far            
            self.n_sample_seen_so_far += 1

    
    def retrieve(self, batch_size):
        if self.cur_idx <= batch_size:
            all_index = torch.arange(self.cur_idx).cuda()
            x, y = self.buffer_img[all_index], self.buffer_label[all_index]
            return x, y
        x,y = None,None
        if self.params.retrieve == "random":
            x,y = Random_Retrieve(self.cur_idx, batch_size, self.buffer_img, self.buffer_label)
        else:
            raise NotImplementedError(
                    'retrieval method not supported: {}'.format(self.params.retrieve))
        return x, y
        
    def update_prototype(self, model):
#        set_trace() #用来调试
        model.eval()
        classes = torch.tensor(list(self.cls_set.keys()))
        #在evaluate.py已经加上model.eval()，作用是不启用 Batch Normalization 和 Dropout
        n = classes.size(0)#因为它与self.n_class是一致的
        
        self.mean_feat = torch.zeros((n, self.feat_size)).float().cuda()
        self.mean_feat_label = torch.zeros(n).long().cuda()

        for i in range(n):
            torch.cuda.empty_cache()
            #获取所有和classes[i]具有相同标签的在buffer_label中的索引
            idx = (self.buffer_label == classes[i]).nonzero(as_tuple=False).flatten()
            #将当前标签加入均值向量标签
            self.mean_feat_label[i] = classes[i]
            #calculate each class' mean feature
            all_img = self.buffer_img[idx]

            #batch 计算方法, 比单张图片依次计算快一些,代码量小
            with torch.no_grad():
                if torch.cuda.is_available():
                    all_img = all_img.cuda() 
                
                all_feat = model.features(all_img)
                all_feat = nn.functional.normalize(all_feat, p=2, dim=1)
                self.mean_feat[i] = all_feat.sum(0) / len(idx) 
                self.mean_feat[i] = nn.functional.normalize(self.mean_feat[i],2,0) 
                          
        model.train()

    def classify(self, x_test, y_test=None):
        
        test_size = x_test.size(0)
        #init predict result
        pred = torch.LongTensor(test_size).fill_(0).cuda()
        #NCM main process 
        for i in range(test_size):
            dis = 100.0            
            for j in range(self.n_class_seen):
                cur_dis = single_euclidean_dist(x_test[i], self.mean_feat[j])                
                if cur_dis < dis:
                    pred[i] = self.mean_feat_label[j]
                    dis = cur_dis 
        return pred
    
    #矩阵计算两两欧式距离,x_test是模型的输出后的feature
    def classify_mat(self, x_test, y_test=None):
        
        dist = euclidean_dist(self.mean_feat, x_test)
        m_idx = torch.argmin(dist, 0)
        pre = self.mean_feat_label[m_idx]
        return pre
    
#def classify_mat(x_test, mean_feat, mean_feat_label):
#
#        dist = euclidean_dist(mean_feat, x_test)
#        m_idx = torch.argmin(dist, 0)
#        pre = mean_feat_label[m_idx]
#        return pre

if __name__ == "__main__":
    initial(0)
#    
    run_start = time.time()
    n = 5000
    classes = 100
    logit = torch.rand((n, classes)).float().cuda()
    label = torch.randint(0, classes, (n,)).cuda()
    prob = nn.functional.softmax(logit, dim=1)
    one_hot = nn.functional.one_hot(label, num_classes=logit.size(1)).bool().cuda()
    tar_max = prob[one_hot]
    prob = prob * ((~one_hot).float())
    ntar_max, _ = torch.max(prob, dim=1)
    score = tar_max - ntar_max
    print(score)
    run_end = time.time()
    print("runing time: ", run_end - run_start)
    

    
    