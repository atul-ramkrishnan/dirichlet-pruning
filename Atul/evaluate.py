import time
import numpy as np
import torch
from util import AverageMeter


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def evaluate(val_loader, model, criterion, print_freq, device):
    """
    Run evaluation
    """
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for i, (input, target) in enumerate(val_loader):
        input = input.to(device)
        target = target.to(device)


        # compute output
        with torch.no_grad():
            output = model(input)
            loss = criterion(output, target)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))


        if i % print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader), loss=losses,
                      top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg


def evaluate_switch_at_layer(val_loader, model, layer, device):
    # print('Prediction when network is forced to predict')
    model.eval()
    correct = 0
    total = 0
    for j, data in enumerate(val_loader):
        images, labels = data
        images = images.to(device)
        #dummy works as it should, if we don't execute switch function in forward the accuracy should be original, 99.27
        #predicted_prob = model.forward(images, "dummy")[0]  # if using switches
        #predicted_prob = model.forward(images, "c1")[0] #13.68 for 99.27
        #predicted_prob = model.forward(images, "c3")[0] #11.35
        with torch.no_grad():
            predicted_prob = model.forward(images, layer)[0]
        predicted = np.argmax(predicted_prob.cpu().detach().numpy(), axis=1)
        total += labels.size(0)
        correct += (predicted == labels.numpy()).sum().item()
    accuracy = 100 * float(correct) / total
    return accuracy

def get_test_accuracy(print_freq, device):
    pass