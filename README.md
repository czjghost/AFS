# AFS (Adaptive Focus Shifting algorithm)
Official implementation of the paper [[New Insights on Relieving Task-Recency Bias for Online Class Incremental Learning]](https://arxiv.org/abs/2302.08243).

We will upload our implementation code as soon as possible (possibly after our paper is published). The backbone of project mainly refers to [online-continual-learning](https://github.com/RaptorMai/online-continual-learning). 

Our method can be easily reimplement from [online-continual-learning
](https://github.com/RaptorMai/online-continual-learning) and [Teacher-free-Knowledge-Distillation
](https://github.com/yuanli2333/Teacher-free-Knowledge-Distillation). We will offer the hyperparameter details synchronously when we upload our implementation code.

## Requirements
Create a virtual enviroment **sh virtualenv online-cl**

Activating a virtual environment **sh source online-cl/bin/activate**

Installing packages **sh pip install -r requirements.txt**

## Datasets 

### Online Class Incremental
- Split CIFAR10
- Split CIFAR100
- Split Mini-ImageNet

### Data preparation
- CIFAR10 & CIFAR100 will be downloaded during the first run. (continuums/datasets/cifar10; continuums/datasets/cifar100)
- Mini-ImageNet: Download from https://www.kaggle.com/whitemoon/miniimagenet/download , and place it in **continuums/datasets/mini_imagenet/**

## Compared methods 
Except our implementation code, you could easily find other implementation results from [SCR](https://github.com/RaptorMai/online-continual-learning), [DVC](https://github.com/YananGu/DVC), [ER-ACE](https://github.com/pclucas14/AML) and [OCM](https://github.com/gydpku/OCM). 

* AGEM: Averaged Gradient Episodic Memory (**ICLR, 2019**) [[Paper]](https://openreview.net/forum?id=Hkf2_sC5FX)
* ER: Experience Replay (**ICML Workshop, 2019**) [[Paper]](https://arxiv.org/abs/1902.10486)
* MIR: Maximally Interfered Retrieval (**NeurIPS, 2019**) [[Paper]](https://proceedings.neurips.cc/paper/2019/hash/15825aee15eb335cc13f9b559f166ee8-Abstract.html)
* GSS: Gradient-Based Sample Selection (**NeurIPS, 2019**) [[Paper]](https://arxiv.org/pdf/1903.08671.pdf)
* GDumb: Greedy Sampler and Dumb Learner (**ECCV, 2020**) [[Paper]](https://www.robots.ox.ac.uk/~tvg/publications/2020/gdumb.pdf)
* ASER: Adversarial Shapley Value Experience Replay (**AAAI, 2021**) [[Paper]](https://arxiv.org/abs/2009.00093)
* SCR: Supervised Contrastive Replay (**CVPR Workshop, 2021**) [[Paper]](https://arxiv.org/abs/2103.13885) 
* DVC: Dual View Consistency (**CVPR, 2022**) [[Paper]](https://openaccess.thecvf.com/content/CVPR2022/html/Gu_Not_Just_Selection_but_Exploration_Online_Class-Incremental_Continual_Learning_via_CVPR_2022_paper.html)
* OCM: Online Continual learning based on Mutual information maximization (**ICML, 2022**) [[Paper]](https://proceedings.mlr.press/v162/guo22g/guo22g.pdf)
* ER-ACE: Cross-Entropy based Alternative (**ICLR, 2022**) [[Paper]](https://openreview.net/pdf?id=N8MaByOzUfb)


## Tricks
- In our paper, the main trick (RV) can be found in BER [[Paper]](https://arxiv.org/abs/2007.05683) and online CL survey [[Paper]](https://arxiv.org/pdf/2101.10423.pdf) which has been implemented by [RaptorMai](https://github.com/RaptorMai/online-continual-learning). 

## Run commands
Detailed descriptions of options can be found in [general_main.py](general_main.py). This file is not uploaded currently.

### Command for duplicate results
We will upload them as soon as possible.

#### CIFAR-10
Memory = 0.2k
```shell
  python general_main.py --agent er --loss rfocal --classify max --data cifar10 --eps_mem_batch 100 --mem_size 200 --review_trick True --kd_trick True --kd_lamda 0.05 --cor_prob 0.99 --T 20.0 --fix_order True
 ```
Memory = 0.5k
```shell
  python general_main.py --agent er --loss rfocal --classify max --data cifar10 --eps_mem_batch 100 --mem_size 500 --review_trick True --kd_trick True --kd_lamda 0.05 --cor_prob 0.99 --T 20.0 --fix_order True
 ```
Memory = 1k
```shell
  python general_main.py --agent er --loss rfocal --classify max --data cifar10 --eps_mem_batch 100 --mem_size 1000 --review_trick True --kd_trick True --kd_lamda 0.1 --cor_prob 0.99 --T 20.0 --fix_order True
 ```
#### CIFAR-100
Memory = 1k
```shell
  python general_main.py --agent er --loss rfocal --classify max --data cifar100 --eps_mem_batch 100 --mem_size 1000 --review_trick True --kd_trick True --kd_lamda 0.15 --cor_prob 0.99 --T 20.0 --fix_order True
 ```
Memory = 2k
```shell
  python general_main.py --agent er --loss rfocal --classify max --data cifar100 --eps_mem_batch 100 --mem_size 2000 --review_trick True --kd_trick True --kd_lamda 0.05 --cor_prob 0.99 --T 20.0 --fix_order True
 ```
Memory = 5k
```shell
  python general_main.py --agent er --loss rfocal --classify max --data cifar100 --eps_mem_batch 100 --mem_size 5000 --review_trick True --kd_trick True --kd_lamda 0.1 --cor_prob 0.99 --T 20.0 --fix_order True
 ```

#### Mini-Imagenet
Memory = 1k
```shell
  python general_main.py --agent er --loss rfocal --classify max --data mini_imagenet --eps_mem_batch 100 --mem_size 1000 --review_trick True --kd_trick True --kd_lamda 0.05 --cor_prob 0.99 --T 20.0 --fix_order True
 ```
Memory = 2k
```shell
  python general_main.py --agent er --loss rfocal --classify max --data mini_imagenet --eps_mem_batch 100 --mem_size 2000 --review_trick True --kd_trick True --kd_lamda 0.1 --cor_prob 0.99 --T 20.0 --fix_order True
 ```
Memory = 5k
```shell
  python general_main.py --agent er --loss rfocal --classify max --data mini_imagenet --eps_mem_batch 100 --mem_size 5000 --review_trick True --kd_trick True --kd_lamda 0.05 --cor_prob 0.99 --T 20.0 --fix_order True
 ```

## Reference

Thanks [RaptorMai](https://github.com/RaptorMai) for selflessly sharing his implementation about recent state-of-the-art methods.
