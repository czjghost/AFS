import torch
from models.resnet import Reduced_ResNet18
from torchvision import transforms
import torch.nn as nn
from loss.FocalLoss import FocalLoss
from loss.RevisedFocalLoss import RevisedFocalLoss
from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomGrayscale, RandomVerticalFlip, RandomRotation

input_size_match = {
    'cifar100': [3, 32, 32],
    'cifar10': [3, 32, 32],
    'mini_imagenet': [3, 84, 84]
}

n_classes = {
    'cifar100': 100,
    'cifar10': 10,
    'mini_imagenet': 100,
}

feature_size_match = {
    'cifar100': 160,
    'cifar10': 160,
    'mini_imagenet': 640,
}

transforms_match = {
    'cifar100': transforms.Compose([
        transforms.ToTensor(),
        ]),
    'cifar10': transforms.Compose([
        transforms.ToTensor(),
        ]),
    'mini_imagenet': transforms.Compose([
        transforms.ToTensor()]),
}

def setup_architecture(params):
    nclass = n_classes[params.data]
    if params.data == 'cifar100':
        model = Reduced_ResNet18(nclass)
        return model
    elif params.data == 'cifar10':
        model = Reduced_ResNet18(nclass)
        return model
    elif params.data == 'mini_imagenet':
        model = Reduced_ResNet18(nclass)
        model.linear = nn.Linear(640, nclass, bias=True)
        return model
    else:
        Exception('wrong dataset name')

def setup_opt(optimizer, model, lr, wd):
    if optimizer == 'SGD':
        optim = torch.optim.SGD(model.parameters(),
                                lr=lr,
                                weight_decay=wd)
    elif optimizer == 'Adam':
        optim = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 weight_decay=wd)
    else:
        raise Exception('wrong optimizer name')
    return optim

def setup_crit(params):
    if params.loss == "focal":
        criterion = FocalLoss(params.focal_alpha, params.focal_gamma)
    elif params.loss == "ce":#包含了softmax
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    elif params.loss == "rfocal":
        criterion = RevisedFocalLoss(params.rfocal_alpha, params.rfocal_sigma, params.rfocal_miu)
    else:
        raise NotImplementedError(
                'loss not supported: {}'.format(params.loss))
    return criterion

def setup_augment(params):
    aug_transform = nn.Sequential(
            RandomResizedCrop(size=(input_size_match[params.data][1], input_size_match[params.data][2]), scale=(0.75, 1.)),
            RandomHorizontalFlip(),
            ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
    )
    if params.data == 'cifar10':
        aug_transform.add_module('3', RandomGrayscale(p=0.2))
        
    return aug_transform