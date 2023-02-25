import torch
import os

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

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


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)


def prepare_for_training(save_dir, device):
    # Check the save_dir exists or not
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if device == torch.device('cpu'):
        print("Running on CPU")
    else:
        print("Running on GPU")
