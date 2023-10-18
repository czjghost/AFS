# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 18:44:21 2022

@author: czjghost
"""
import torch
from torch import nn 
from torch.nn import functional as F
import math
import numpy as np
import random
from copy import deepcopy

class kd_manager:
    def __init__(self, params):
        self.params = params
        self.T = params.T
#        self.teacher_model = None
        self.vkd = Virtual_KD(params)
        self.lamda = params.kd_lamda
    
    def get_loss(self, x, y, model):
        logits = model(x)
        loss = self.vkd(y, logits)
        loss = loss * (self.T ** 2)
        loss = loss * self.lamda
        
        return loss

class Virtual_KD(nn.Module):
    def __init__(self, params):        
        super(Virtual_KD, self).__init__()
        self.T = params.T
        self.cor_prob = params.cor_prob
        self.K = params.num_classes
        
    def forward(self, targets, logits):
        log_scores_norm = F.log_softmax(logits / self.T, dim=1)
        #generate virtual teacher’s output
        old_logits = torch.ones_like(logits) * (1 - self.cor_prob) / (self.K - 1.0)
        old_logits = old_logits.cuda()
        for i in range(logits.shape[0]):
            old_logits[i ,targets[i]] = self.cor_prob
        
        targets_norm = F.softmax(old_logits / self.T, dim=1)
        # Calculate distillation loss (see e.g., Li and Hoiem, 2017)
        kd_loss = (-1.0 * targets_norm * log_scores_norm).sum(dim=1).mean()
        return kd_loss


def initial(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

if __name__ == "__main__":
    initial(0)
#    n = 11
#    crit = Virtual_KD(2.0, 0.8, n)
#    logits = torch.randn((n, n))
#    targets = torch.randint(0,n,(n,))
#    print(targets)
#    loss = crit(targets, logits)
#    print(loss)
    n = 3
    logits = torch.tensor([[0.01,0.5,0.03],
                           [0.7,0.2,0.03],
                           ]).cuda()
    
    targets = torch.tensor([0,1])
    old_logits = torch.tensor([[0.01,0.5,0.03],
                           [0.7,0.2,0.03],
                           ]).cuda()
    
#    crit = Mean_KD(1.0, n)
#    crit.update(targets, logits)
#    targets = torch.tensor([1,2])
#    crit.update(targets, logits)
#    print(crit.dist_logit)
#    print(crit.seen_samples)
    
    
#    logits = torch.tensor([[0.01,0.5,0.03],
#                           [0.7,0.2,0.03],
#                           [0.7,0.2,0.03],
#                           [0.7,0.2,0.03],
#                           [0.7,0.2,0.03],
#                           ]).cuda()
#    
#    targets = torch.tensor([0,1,2,0,1])
#    loss = crit(targets, logits)
#    print(loss)
    
#    targets = torch.tensor([1,1])
#    print(logits is old_logits)
#    crit = Partial_KD(20.0, n)
#    print(crit(targets, logits, old_logits))
#    crit = Virtual_KD(20.0, 0.8, n)
#    print(crit(targets, logits))
    
    