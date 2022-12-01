# -*- coding: utf-8 -*-
import numpy as np
import torch
import random



def is_cuda_available():
    return torch.cuda.is_available()


def get_device(cuda_number=0):
    """cuda_number : set the number of CUDA, default 0"""

    return torch.device('cpu')#('cuda:{}'.format(cuda_number) if torch.cuda.is_available() else 'cpu')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    pass