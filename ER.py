# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 11:09:04 2022

@author: czjghost
"""

import torch
import os
import numpy as np
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import random
import sys
from continuums.data_utils import transforms_match, dataset_transform, setup_test_loader
#from continuums.cifar10 import CIFAR10
#from continuums.cifar100 import CIFAR100
#from continuums.mini_imagenet import Mini_ImageNet
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib
from models.resnet import Reduced_ResNet18, ResNet18
from continuums.continuum import continuum
from buffer.Reservoir_Random import Reservoir_Random
from loss.FocalLoss import FocalLoss
from loss.RevisedFocalLoss import RevisedFocalLoss
from loss.kd_manager import kd_manager
from copy import deepcopy
import argparse
import math
from torch.utils.data import TensorDataset, Dataset, DataLoader
from setup_elements import setup_architecture, setup_opt, setup_crit, setup_augment
from AverageMeter import AverageMeter
from ipdb import set_trace
import torch.nn.init as init
from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomGrayscale, RandomVerticalFlip
from logger import Logger, build_er_holder
from evaluate import evaluate
from metrics import compute_performance
import time

def l2_norm(x, axit=1):
    norm = torch.norm(x, 2, axit, True)
    output = torch.div(x, norm)
    return output

def get_old_sample(mem_x, mem_y, old_class_label):
    n = mem_y.size(0)
    idx = torch.tensor([]).long()
    for i in range(n):
        if mem_y[i] in old_class_label:
            idx = torch.cat([idx, torch.tensor([i])])
    x,y = mem_x[idx], mem_y[idx]
    return x,y

def experience_replay(args, holder, log):
#    set_trace() #用来调试
    sys.stdout = log
    data_continuum = continuum(args.data, args)
    acc_list_all = []
    
    print(args)
    start = time.time()
    for run_time in range(args.run_time):
        print(args.data + "_mem_size=" + str(args.mem_size) + "_run_time=" + str(run_time))
        
        model = setup_architecture(args)

        aug_transform = setup_augment(args)
        optimizer = setup_opt(args.optimizer, model, args.learning_rate, args.weight_decay)
        criterion = setup_crit(args)
        
        sampler = Reservoir_Random(args) 
        
        if torch.cuda.is_available():
            model = model.cuda()

        old_class_label = torch.tensor([]).long().cuda()        
        kd_criterion = kd_manager(args)
            
        data_continuum.new_run()
        all_test = data_continuum.test_data()#取出测试集
        test_loaders = setup_test_loader(all_test, args)

        acc_list = []
        losses_batch = AverageMeter()
        losses_mem = AverageMeter()

        #关于backward一次就step的一次的区别和backward两次再step, 作者认为后者好一些, 由于batchnorm的原因分开算好
        #https://github.com/RaptorMai/online-continual-learning/issues/9
        ##############################singe dataset CIL training stage##############################
        for task_id, (x_train, y_train, labels) in enumerate(data_continuum):
            print('==>>> task id: {},  {}, {},'.format(task_id, x_train.shape, y_train.shape))
            train_dataset = dataset_transform(x_train, y_train, transform=transforms_match[args.data])
            train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=0,
                                           drop_last=True)
            new_class_label = torch.tensor(labels).long().cuda()

            ##############################singe task training stage##############################
            model.train()
            for batch_id, batch_data in enumerate(train_loader):
                batch_x, batch_y = batch_data
                if torch.cuda.is_available():
                    batch_x = batch_x.cuda()
                    batch_y = batch_y.cuda()

                optimizer.zero_grad()
                x_feat = model.features(batch_x)
                x_logit = model.logits(x_feat)
                
                loss = criterion(x_logit, batch_y)   
                
                if args.kd_trick:
                    loss = loss + kd_criterion.get_loss(batch_x, batch_y, model)
                
                loss.backward()
                #then do not .step(), accumulate the gradient with memory loss for .step() 
                losses_batch.update(loss.item(), batch_y.size(0))

                if not sampler.is_empty():
                    mem_x,mem_y = sampler.retrieve(args.eps_mem_batch)
                        
                    if torch.cuda.is_available():
                        mem_x = mem_x.cuda()
                        mem_y = mem_y.cuda()
                        

                    mem_x = torch.cat([mem_x, aug_transform(mem_x)])
                    mem_y = torch.cat([mem_y, mem_y])
                    
                    #here we do not need to call .zero_grad() because we want to accumulate the gradient
                    mem_x_feat = model.features(mem_x)
                    mem_x_logit = model.logits(mem_x_feat)

                    mem_loss = criterion(mem_x_logit, mem_y)
                    
                    if args.kd_trick:
                        mem_loss = mem_loss + kd_criterion.get_loss(mem_x, mem_y, model)
                        
                    mem_loss.backward()
                    #not .step() here, call .step() outside of if statement
                    losses_mem.update(mem_loss.item(), mem_y.size(0))
                
                #the gradient have accumulated, just call .step()
                optimizer.step()
                
                batch_x, batch_y = batch_data
                sampler.update(batch_x, batch_y)
                if batch_id % 100 == 1:
                            print(
                                '==>>> it: {}, avg loss: {:.6f}, avg mem loss: {:.6f}'
                                    .format(batch_id, losses_batch.avg(), losses_mem.avg()) )

            ##############################after train review trick##############################
            if args.review_trick:
                mem_x = sampler.buffer_img[:sampler.cur_idx]
                mem_y = sampler.buffer_label[:sampler.cur_idx]
                if mem_x.size(0) > 0:
                    rv_dataset = TensorDataset(mem_x, mem_y)
                    rv_loader = DataLoader(rv_dataset, batch_size=args.rv_batch, shuffle=True, 
                                           num_workers=0, drop_last=False)
                    for ep in range(1):
                        for i, batch_data in enumerate(rv_loader):
                            batch_x, batch_y = batch_data
                            if torch.cuda.is_available():
                                batch_x = batch_x.cuda()
                                batch_y = batch_y.cuda()
                            
                            x_logit = model(batch_x)
                                
                            loss = criterion(x_logit, batch_y)  

                            optimizer.zero_grad()
                            loss.backward()
                            #down gradient, same as learning rate / 10.0
                            params = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
                            grad = [p.grad.clone() / 10.0 for p in params]
                            for g, p in zip(grad, params):
                                p.grad.data.copy_(g)
                            optimizer.step() 
                        
            task_acc = evaluate(model, test_loaders, sampler, args)
            
            old_class_label = torch.cat([old_class_label, new_class_label])
            
            #训练完成task_id后,在前task_id上的测试集分别计算准确率,最终是个下三角矩阵
            acc_list.append(task_acc)
        
        #add the result of each run time to acc_list_all
        acc_list_all.append(np.array(acc_list))
        ##############################print acc result##############################
        #print all results for each run
        print("\n----------run {} result-------------".format(run_time))
        for acc in acc_list:    
            print(acc)
        print("last task avr acc: ", np.mean(acc_list[len(acc_list)-1]))
        
        ##############################save acc result##############################
        txt = holder + "/run_time = %d" % (run_time) + ".txt"
        with open(txt, "w") as f:
            for acc in acc_list:    
                f.write(str(list(acc)) + "\n")
            f.write("last task avr acc: %lf" % np.mean(acc_list[len(acc_list)-1]) + "\n")
        
        ##############################save setting parameter##############################
        if not os.path.exists(holder + '/setting.txt'):
            argsDict = args.__dict__
            with open(holder + '/setting.txt', 'w') as f:
                f.writelines('------------------ start ------------------' + '\n')
                for eachArg, value in argsDict.items():
                    f.writelines(eachArg + ' : ' + str(value) + '\n')
                f.writelines('------------------- end -------------------')
        print("\n\n")
        
    ##############################calculate avr result after args.run_time running##############################
    end = time.time()
    avg_end_acc, avg_end_fgt, avg_acc, avg_bwtp, avg_fwt = compute_performance(np.array(acc_list_all))
    with open(holder + '/avr_end_result.txt', 'w') as f:
        f.write('Total run (second):{}\n'.format(end - start))
        f.write('Avg_End_Acc:{}\n'.format(avg_end_acc))
        f.write('Avg_End_Fgt:{}\n'.format(avg_end_fgt))
        f.write('Avg_Acc:{}\n'.format(avg_acc))
        f.write('Avg_Bwtp:{}\n'.format(avg_bwtp))
        f.write('Avg_Fwt:{}\n'.format(avg_fwt))
        
    print('----------- final average result -----------'
              .format(avg_end_acc, avg_end_fgt, avg_acc, avg_bwtp, avg_fwt))        
    print(' Avg_End_Acc {}\n Avg_End_Fgt {}\n Avg_Acc {}\n Avg_Bwtp {}\n Avg_Fwt {}\n'
              .format(avg_end_acc, avg_end_fgt, avg_acc, avg_bwtp, avg_fwt))
