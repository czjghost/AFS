#!/usr/bin/env python
# -*- coding: utf-8 -*-
# other imports
import os
import sys
currentUrl = os.path.dirname(__file__)  
parentUrl = os.path.abspath(os.path.join(currentUrl, os.pardir))  
sys.path.append(parentUrl)
from continuums.cifar10 import CIFAR10
from continuums.cifar100 import CIFAR100
from continuums.mini_imagenet import Mini_ImageNet

data_objects = {
    'cifar100': CIFAR100,
    'cifar10': CIFAR10,
    'mini_imagenet': Mini_ImageNet,
}

class continuum(object):
    def __init__(self, dataset, params):
        """" Initialize Object """
        self.data_object = data_objects[dataset](params)
#        self.run = params.run_time
        self.task_nums = self.data_object.task_nums
        self.cur_task = 0
        self.cur_run = -1

    def __iter__(self):
        return self

    def __next__(self):
        if self.cur_task == self.data_object.task_nums:
            raise StopIteration
        x_train, y_train, labels = self.data_object.new_task(self.cur_task, cur_run=self.cur_run)
        self.cur_task += 1
        return x_train, y_train, labels
    
    #在基类DataBase中已经定义过了,cifar10/cifar100/miniimagenet里面不再定义
    def test_data(self):
        return self.data_object.get_test_set()

    def clean_mem_test_set(self):
        self.data_object.clean_mem_test_set()

    def reset_run(self):
        self.cur_task = 0

    def new_run(self):
        self.cur_task = 0
        self.cur_run += 1
        self.data_object.new_run(cur_run=self.cur_run)


