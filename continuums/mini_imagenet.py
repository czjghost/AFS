import pickle
import numpy as np
import os
import sys
currentUrl = os.path.dirname(__file__)  
parentUrl = os.path.abspath(os.path.join(currentUrl, os.pardir))  
sys.path.append(parentUrl)
from continuums.data_utils import create_task_composition, load_task_with_labels, shuffle_data
from continuums.dataset_base import DatasetBase
import argparse
import torch

TEST_SPLIT = 1 / 6

class Mini_ImageNet(DatasetBase):
    def __init__(self, params):
        dataset = 'mini_imagenet'
        num_tasks = params.num_tasks
        super(Mini_ImageNet, self).__init__(dataset, num_tasks, params)


    def download_load(self):
        train_in = open("continuums/datasets/mini_imagenet/mini-imagenet-cache-train.pkl", "rb")
        train = pickle.load(train_in)
        train_x = train["image_data"].reshape([64, 600, 84, 84, 3])
        val_in = open("continuums/datasets/mini_imagenet/mini-imagenet-cache-val.pkl", "rb")
        val = pickle.load(val_in)
        val_x = val['image_data'].reshape([16, 600, 84, 84, 3])
        test_in = open("continuums/datasets/mini_imagenet/mini-imagenet-cache-test.pkl", "rb")
        test = pickle.load(test_in)
        test_x = test['image_data'].reshape([20, 600, 84, 84, 3])
        all_data = np.vstack((train_x, val_x, test_x))
        train_data = []
        train_label = []
        test_data = []
        test_label = []
        for i in range(len(all_data)):
            cur_x = all_data[i]
            cur_y = np.ones((600,)) * i
            rdm_x, rdm_y = shuffle_data(cur_x, cur_y)
            x_test = rdm_x[: int(600 * TEST_SPLIT)]
            y_test = rdm_y[: int(600 * TEST_SPLIT)]
            x_train = rdm_x[int(600 * TEST_SPLIT):]
            y_train = rdm_y[int(600 * TEST_SPLIT):]
            train_data.append(x_train)
            train_label.append(y_train)
            test_data.append(x_test)
            test_label.append(y_test)
        self.train_data = np.concatenate(train_data)
        self.train_label = np.concatenate(train_label)
        self.test_data = np.concatenate(test_data)
        self.test_label = np.concatenate(test_label)

    def new_run(self, **kwargs):
        self.setup()
        return self.test_set

    def new_task(self, cur_task, **kwargs):
        labels = self.task_labels[cur_task]
        x_train, y_train = load_task_with_labels(self.train_data, self.train_label, labels)
        return x_train, y_train, labels

    def setup(self):
        self.task_labels = create_task_composition(class_nums=100, num_tasks=self.task_nums,
                                                       fixed_order=self.params.fix_order)
        self.test_set = []
        for labels in self.task_labels:
            x_test, y_test = load_task_with_labels(self.test_data, self.test_label, labels)
            self.test_set.append((x_test, y_test))


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
    
    parser.add_argument('--num_tasks', dest='num_tasks', default=10,
                        type=int,
                        help='Number of tasks (default: %(default)s), OpenLORIS num_tasks is predefined')
    #是否固定顺序
    parser.add_argument('--fix_order', dest='fix_order', default=False,
                        type=bool,
                        help='In NC scenario, should the class order be fixed (default: %(default)s)')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    
    data = Mini_ImageNet(args)
    data.setup()
    x_train, y_train, labels = data.new_task(0)
    print(x_train.shape)