import torch
import os
from enum import Enum


class Method(Enum):
    DIRICHLET = 1
    GENERALIZED_DIRICHLET = 2


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def restore(self, val, avg, sum, count):
        self.val = val
        self.avg = avg
        self.sum = sum
        self.count = count
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)


def create_dir_if_not_exists(save_dir):
    # Check the save_dir exists or not
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
