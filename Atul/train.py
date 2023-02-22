import os
import time
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import vgg
from dataloader import get_train_valid_loader
from util import AverageMeter, save_checkpoint, prepare_for_training, accuracy
from evaluate import evaluate


def adjust_learning_rate(optimizer, lr, epoch):
    """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
    lr = lr * (0.5 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_one_epoch(train_loader, model, criterion, optimizer, epoch, print_freq, device):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        input = input.to(device)
        target = target.to(device)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))



def train(model_type, save_dir, cpu, resume, eval, batch_size, workers, lr, momentum, weight_decay, start_epoch, epochs, print_freq):
    prepare_for_training(save_dir, cpu)
    
    if cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
    
    # Only support VGG16_BN for now
    model = vgg.vgg16_bn()
    model.to(device)

    best_prec1 = 0
    if resume:
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(eval, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume))

    train_loader, val_loader = get_train_valid_loader('./dataset',
                                                    batch_size,
                                                    augment=True,
                                                    random_seed=0,
                                                    num_workers=workers)

    criterion = nn.CrossEntropyLoss()
    criterion.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr,
                                momentum=momentum,
                                weight_decay=weight_decay)

    
    for epoch in range(start_epoch, epochs):
        adjust_learning_rate(optimizer, lr, epoch)

        # train for one epoch
        train_one_epoch(train_loader, model, criterion, optimizer, epoch, print_freq, device)

        # evaluate on validation set
        prec1 = evaluate(val_loader, model, criterion, print_freq, device)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=os.path.join(save_dir, 'checkpoint_{}.tar'.format(epoch)))