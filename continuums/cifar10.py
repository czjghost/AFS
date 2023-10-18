import numpy as np
import os
from torchvision import datasets
import sys
currentUrl = os.path.dirname(__file__)  
parentUrl = os.path.abspath(os.path.join(currentUrl, os.pardir))  
sys.path.append(parentUrl)
from continuums.data_utils import create_task_composition, load_task_with_labels, shuffle_data
from continuums.dataset_base import DatasetBase
import argparse
import torch

class CIFAR10(DatasetBase):
    def __init__(self, params):
        dataset = 'cifar10'

        num_tasks = params.num_tasks
        super(CIFAR10, self).__init__(dataset, num_tasks, params)

    def download_load(self):
        dataset_train = datasets.CIFAR10(root=self.root, train=True, download=True)
        self.train_data = dataset_train.data
        self.train_label = np.array(dataset_train.targets)
        dataset_test = datasets.CIFAR10(root=self.root, train=False, download=True)
        self.test_data = dataset_test.data
        self.test_label = np.array(dataset_test.targets)

    def setup(self):#首先进行导入和任务分类，在定义该类之后就应该立马setup，在这里作者用new_run并返回测试集
        self.task_labels = create_task_composition(class_nums=10, num_tasks=self.task_nums, fixed_order=self.params.fix_order)
        self.test_set = []
        for labels in self.task_labels:
            x_test, y_test = load_task_with_labels(self.test_data, self.test_label, labels)
            self.test_set.append((x_test, y_test))

    def new_task(self, cur_task, **kwargs):#每一个任务调用一次，每一个任务里面分多个batch
        labels = self.task_labels[cur_task]
        x_train, y_train = load_task_with_labels(self.train_data, self.train_label, labels)
        #x_train,y_train分别对应标签,labels对应当前任务含的类标签(去重后的)
        return x_train, y_train, labels

    def new_run(self, **kwargs):#第一次运行 先调用这个
        self.setup()
        return self.test_set


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Online Continual Learning PyTorch")
    #验证集比例
    parser.add_argument('--num_runs', dest='num_runs', default=1, type=int,
                        help='Number of runs (default: %(default)s)')
    parser.add_argument('--val_size', dest='val_size', default=0.0, type=float,
                        help='val_size (default: %(default)s)')
    #用于验证的批次数 
    parser.add_argument('--num_val', dest='num_val', default=3, type=int,
                        help='Number of batches used for validation (default: %(default)s)')
    #验证运行的次数
    parser.add_argument('--num_runs_val', dest='num_runs_val', default=3, type=int,
                        help='Number of runs for validation (default: %(default)s)')
    
    parser.add_argument('--num_tasks', dest='num_tasks', default=5,
                        type=int,
                        help='Number of tasks (default: %(default)s), OpenLORIS num_tasks is predefined')
    #是否固定顺序
    parser.add_argument('--fix_order', dest='fix_order', default=False,
                        type=bool,
                        help='In NC scenario, should the class order be fixed (default: %(default)s)')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    
#    data = CIFAR10(args)
#    data.setup()
#    x_train, y_train, labels = data.new_task(0)
#    print(x_train.shape)
    