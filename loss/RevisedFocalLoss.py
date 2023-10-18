# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 22:42:01 2022

@author: czjghost
"""

import torch
import torch.nn as nn
import random
import numpy as np

def initial(seed):
    #https://blog.csdn.net/weixin_41990278/article/details/106268969
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

class RevisedFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, sigma=0.5, miu=0.3):
        """
        :param alpha_t: A list of weights for each class
        :param gamma:
        """
        super(RevisedFocalLoss, self).__init__()
        self.alpha = alpha
        self.sigma = sigma
        self.miu = miu

    def forward(self, outputs, targets, new_offset=None, reduce="mean"):
        # important to add reduction='none' to keep per-batch-item loss
        ce_loss = torch.nn.functional.cross_entropy(outputs, targets, reduction='none') 
        
        pt = torch.exp(-ce_loss)
        
        weight = 0
        if new_offset is None:
            weight = torch.exp(- (pt - self.miu) ** 2 / self.sigma)
        else:
            offset = new_offset[targets]
            weight = torch.exp(- (pt - offset) ** 2 / self.sigma)
#        print(beta)

        rfocal_loss =  (self.alpha * weight * ce_loss)
        
        if reduce=="mean":
            rfocal_loss = rfocal_loss.mean() # mean over the batch
        elif reduce == "sum":
            rfocal_loss = rfocal_loss.sum()
            
        return rfocal_loss


if __name__ == '__main__':
    initial(0)
    outputs = torch.tensor([[1., 1., 1.],
                            [2.5, 7.5, 5.0],
                            [4., 2., 6.],
                            [2., 1., 4.],
                            ])
    
    targets = torch.tensor([0, 2, 0, 1])
    new_offset = torch.tensor([0.1, 0.2, 1.0])
#    print(torch.nn.functional.softmax(outputs, dim=1))
#
    fl= RevisedFocalLoss()
##    
    loss = fl(outputs, targets, new_offset)
    print(loss)
#    
#    loss1 = nn.functional.cross_entropy(outputs, targets)
#    print(loss1.item())
#    print(2.0 ** 4 / 2)