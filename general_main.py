import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import sys
from ER import experience_replay
from logger import Logger, build_er_holder
import matplotlib
matplotlib.use('Agg')

n_classes = {
    'cifar100': 100,
    'cifar10': 10,
    'mini_imagenet': 100,
}

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def parameter_setting():
    # Commandline arguments
    parser = argparse.ArgumentParser(description="Online Continual Learning PyTorch")
    ########################General#########################
    parser.add_argument('--seed', dest='seed', default=0, type=int,
                        help='Random seed, if seed < 0, it will not be set')
    
    #选择的数据集，输入对应的路径, 论文中常用前三种
    parser.add_argument('--data', dest='data', default="cifar10", type=str,
                        choices=['cifar10', 'cifar100', 'mini_imagenet'],
                        help='Path to the dataset. (default: %(default)s)')
    
    parser.add_argument('--agent', dest='agent', default="er", type=str,
                        choices=['er',],
                        help='the agent for continual learning')
    
    parser.add_argument('--retrieve', dest='retrieve', default="random", type=str,
                        choices = ["random",],
                        help='retrieve method: random')
    
    parser.add_argument('--run_time', dest='run_time', default=10, type=int,
                        help='the time of running')

    #2018 ECCV End-to-End Incremental Learning(lr * 0.1), reference from SCR author(gradient / 10) 等价
    #https://github.com/mmasana/FACIL/blob/f653d6c0eef52292dd610f3fd412e29315a93ed2/src/approach/eeil.py#L75
    parser.add_argument('--review_trick', dest='review_trick', default=True, type=boolean_string,
                        help='whethre use review trick')
    
    ########################Mid Focal Loss#########################
    #default : create by czj
    parser.add_argument('--rfocal_alpha', dest='rfocal_alpha', default=0.25, type=float,
                        help='Alpha in Revised Focal loss')
    
    parser.add_argument('--rfocal_sigma', dest='rfocal_sigma', default=0.5, type=float,
                        help='Sigma in Revised Focal loss')
    
    parser.add_argument('--rfocal_miu', dest='rfocal_miu', default=0.3, type=float,
                        help='Miu in Revised Focal loss')
    
    ########################kd loss#########################
    
    parser.add_argument('--kd_trick', dest='kd_trick', default=False, type=boolean_string,
                        help='whethre use knowledge distillation loss')
    
    parser.add_argument('--kd_lamda', dest='kd_lamda', default=0.1, type=float,
                        choices=[0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5],
                        help='lamda used in virtual knowledge distillation loss')
    
    parser.add_argument('--cor_prob', dest='cor_prob', default=0.99, type=float,
                        help='the correct probability of target class in virtual kd')
    
    parser.add_argument('--T', dest='T', default=20.0, type=float,
                        help='temperature for Distillation loss function, paper set to 2 default')
    
    ########################Experience Replay#########################
    parser.add_argument('--loss', dest='loss', default="rfocal", type=str,
                        choices = ["ce","focal","rfocal"],
                        help='select loss')
    
    parser.add_argument('--classify', dest='classify', default="max", type=str,
                        choices = ["ncm", "max"],
                        help='select classification')
    
    ########################Focal Loss#########################
    #default : alpha=0.25, gamma=2.0
    parser.add_argument('--focal_alpha', dest='focal_alpha', default=0.25, type=float,
                        help='Focal loss alpha')
    
    parser.add_argument('--focal_gamma', dest='focal_gamma', default=2.0, type=float,
                        help='Focal loss gamma')
    
    ########################Optimizer#########################
    #优化器
    parser.add_argument('--optimizer', dest='optimizer', default='SGD', type=str,
                        choices=['SGD', 'Adam'],
                        help='Optimizer (default: %(default)s)')
    #学习率
    parser.add_argument('--learning_rate', dest='learning_rate', default=0.1,
                        type=float,
                        help='Learning_rate of models (default: %(default)s)')
    #batch size
    parser.add_argument('--batch', dest='batch', default=10,
                        type=int,
                        help='Batch size (default: %(default)s)')
    #batch size for review trick
    parser.add_argument('--rv_batch', dest='rv_batch', default=10,
                        type=int,
                        help='Batch size for Review trick (default: %(default)s)')
    #每个batch抽取内存中的多少样本进行测试，一般和一个batch一致
    parser.add_argument('--eps_mem_batch', dest='eps_mem_batch', default=100,
                        type=int,
                        help='the number of sample selected per batch from memory (default: %(default)s)')
    #内存大小设置
    parser.add_argument('--mem_size', dest='mem_size', default=1000,
                        type=int,
                        help='Memory buffer size (default: %(default)s)')    
    #测试集的batch size
    parser.add_argument('--test_batch', dest='test_batch', default=128,
                        type=int,
                        help='Test batch size (default: %(default)s)')
    #是否进行权重衰减
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=0,
                        help='weight_decay')

    ########################Data#########################
    #分割的任务数，即总类别数/num_task 就是最终每个任务包含的类别数
    parser.add_argument('--num_tasks', dest='num_tasks', default=5,
                        type=int,
                        help='Number of tasks (default: %(default)s), OpenLORIS num_tasks is predefined')
    
    parser.add_argument('--num_classes', dest='num_classes', default=10,
                        type=int,
                        help='Number of classes in total')
    #是否固定顺序，为了和之前的文章公平比较这里固定了顺序
    parser.add_argument('--fix_order', dest='fix_order', default=True,
                        type=bool,
                        help='In NC scenario, should the class order be fixed (default: %(default)s)')
    
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    
    return args

def multiple_run(args, holder, log):
    if args.seed >= 0:
        initial(args)
    
    if args.data == 'cifar10':
        args.num_tasks = 5
    elif args.data == 'mini_imagenet':
        args.num_tasks = 10 
    elif args.data == 'cifar100':
        args.num_tasks = 10 
    
    args.num_classes = n_classes[args.data]
    
    if args.agent == "er":
        experience_replay(args, holder, log)
    else:
        raise NotImplementedError(
                'agent not supported: {}'.format(args.agent))
                
def initial(args):
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    #做好初始化的准备
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed) 
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    #all config
    args = parameter_setting()

    holder = None
    if args.agent == "er":
        holder = build_er_holder(args)
    else:
        raise NotImplementedError(
                'agent not supported: {}'.format(args.agent))
        
    log = Logger(holder + "/running_result.log",stream=sys.stdout)
    sys.stdout = log
    sys.stderr = log
    
    multiple_run(args, holder, log)
    

