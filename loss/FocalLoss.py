# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 22:14:57 2022

@author: czjghost
"""
import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        """
        :param alpha_t: A list of weights for each class
        :param gamma:
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, outputs, targets, reduce="mean"):
        ce_loss = torch.nn.functional.cross_entropy(outputs, targets, reduction='none') # important to add reduction='none' to keep per-batch-item loss
        
        pt = torch.exp(-ce_loss)
#        print(pt)
        focal_loss = (self.alpha * (1-pt)**self.gamma * ce_loss)
        
        if reduce=="mean":
            focal_loss = focal_loss.mean() # mean over the batch
        elif reduce == "sum":
            focal_loss = focal_loss.sum()
            
        return focal_loss


if __name__ == '__main__':
    outputs = torch.tensor([[2, 1.,3.],
                            [2.5, 1.,5.],
                            [4., 2.,6.]
                            ])
    targets = torch.tensor([0, 1, 2])
#    print(torch.nn.functional.softmax(outputs, dim=1))

    fl= FocalLoss(1, 1)
    
    loss = fl(outputs, targets)
    print(loss.item())
    
    loss1 = nn.functional.cross_entropy(outputs, targets)
    print(loss1.item())