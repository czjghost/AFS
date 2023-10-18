import time
import sys
import os
import random


def loop(pre):
    r = os.system(pre)
    print("return result: ", r)

if __name__ == "__main__":
    c = []
    c.append("CUDA_VISIBLE_DEVICES=0 python general_main.py --agent er --loss rfocal --classify max --data cifar10 --eps_mem_batch 100 --mem_size 200 --review_trick True --kd_trick True --kd_lamda 0.05 --cor_prob 0.99 --T 20.0 --fix_order True")
    c.append("CUDA_VISIBLE_DEVICES=0 python general_main.py --agent er --loss rfocal --classify max --data cifar10 --eps_mem_batch 100 --mem_size 500 --review_trick True --kd_trick True --kd_lamda 0.05 --cor_prob 0.99 --T 20.0 --fix_order True")
    c.append("CUDA_VISIBLE_DEVICES=0 python general_main.py --agent er --loss rfocal --classify max --data cifar10 --eps_mem_batch 100 --mem_size 1000 --review_trick True --kd_trick True --kd_lamda 0.1 --cor_prob 0.99 --T 20.0 --fix_order True")
    
    for each in c:
        loop(each)
    