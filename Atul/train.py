import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import vgg
from dataloader import get_train_valid_loader
from util import AverageMeter, save_checkpoint, create_dir_if_not_exists, Method
from evaluate import evaluate, evaluate_switch_at_layer, accuracy
import shutil


def adjust_learning_rate(optimizer, lr, epoch):
    """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
    lr = lr * (0.5 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_one_epoch(train_loader, model, criterion, optimizer, epoch, print_freq, device):
    """
        Run one train epoch
    """
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    for i, (input, target) in enumerate(train_loader):
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

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), loss=losses, top1=top1))



def train(model_type, save_dir, device, resume, eval, batch_size, workers, lr, momentum, weight_decay, start_epoch, epochs, print_freq):
    file_path = os.path.join(save_dir, 'models')
    create_dir_if_not_exists(file_path)
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

    optimizer = torch.optim.Adam(model.parameters(), lr)

    
    for epoch in range(start_epoch, epochs):
        # adjust_learning_rate(optimizer, lr, epoch)

        # train for one epoch
        train_one_epoch(train_loader, model, criterion, optimizer, epoch, print_freq, device)

        # evaluate on validation set
        prec1 = evaluate(val_loader, model, criterion, print_freq, device)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        if is_best:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optim_state_dict': optimizer.state_dict()
            }, filename=os.path.join(file_path, f'checkpoint_{model_type}_epoch{epoch}.tar'))


def loss_function_dirichlet(prediction, true_y, S, alpha_0, hidden_dim, how_many_samps, annealing_rate, device):
    cross_entropy = nn.functional.cross_entropy(prediction, true_y)
    # KLD term
    alpha_0 = torch.Tensor([alpha_0]).to(device)
    hidden_dim = torch.Tensor([hidden_dim]).to(device)
    trm1 = torch.lgamma(torch.sum(S)) - torch.lgamma(hidden_dim * alpha_0)
    trm2 = - torch.sum(torch.lgamma(S)) + hidden_dim * torch.lgamma(alpha_0)
    trm3 = torch.sum((S - alpha_0) * (torch.digamma(S) - torch.digamma(torch.sum(S))))
    KLD = trm1 + trm2 + trm3
    # annealing kl-divergence term is better

    # if verbose:
    #     print("<----------------------------------------------------------------------------->")
    #     print("Prior alpha", alpha_0)
    #     print("\n")
    #     print("Posterior alpha", S)
    #     print("\n")
    #     print("Prior mean", mean_Dirichlet(alpha_0))
    #     print("Posterior mean", mean_Dirichlet(S))
    #     print("\n")
    #     print("KL divergence", KLD)
    #     print("\n")

    return cross_entropy + annealing_rate * KLD / how_many_samps


def train_one_importance_switch(method, train_loader, val_loader, lr, epochs, start_epoch, layer, switch_samps, device, resume, original, batch_size, print_freq, save_dir, create_bkp):
    print(f"=> Training importance switch at layer {layer}")
    file_path = os.path.join(save_dir, 'models', method)
    create_dir_if_not_exists(file_path)
    if create_bkp:
        file_path_bkp = os.path.join(save_dir, 'models', method, 'backup')
        create_dir_if_not_exists(file_path_bkp)

    if method == "dirichlet":
        method = Method.DIRICHLET
    elif method == "generalized_dirichlet":
        method = Method.GENERALIZED_DIRICHLET

    model = vgg.vgg16_bn(method=method, switch_samps=switch_samps, hidden_dim=layer, device=device).to(device)

    if method == Method.DIRICHLET:
        optimizer = optim.Adam([model.switch_parameter_alpha], lr=lr)

    elif method == Method.GENERALIZED_DIRICHLET:
        pass

    if os.path.isfile(resume):
        print("=> resuming importance switch training")
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        optimizer.load_state_dict(checkpoint['optim_state_dict'])

    elif os.path.isfile(original):
        print("=> importance switch training from scratch")
        print("=> loading original (uncompressed) model '{}'".format(original))
        checkpoint = torch.load(original)
        model.load_state_dict(checkpoint['state_dict'], strict=False)

    else:
        raise Exception("Cannot continue")

    model.train()
    best_accuracy = 0
    how_many_epochs = 200
    annealing_steps = float(8000. * how_many_epochs)
    beta_func = lambda s: min(s, annealing_steps) / annealing_steps

    for epoch in range(start_epoch, epochs):
        print(f"<--------------------------------------Begin Epoch {epoch}-------------------------------------->")
        losses = AverageMeter()
        annealing_rate = beta_func(epoch)
        model.train()
        for i, (input, labels) in enumerate(train_loader):
            if i >= 20:             # Only for testing.
                break
            input, labels = input.to(device), labels.to(device)
            outputs, S = model(input, layer)
            alpha_0 = 2
            loss = loss_function_dirichlet(outputs, labels, S, alpha_0, vgg.vgg16_hidden_dims[layer], batch_size, annealing_rate, device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.update(loss.item(), input.size(0))
            if i % print_freq == 0:
                print(f"Epoch: [{epoch}] [{i}/{len(train_loader)}]\t", end='')
                print(f"Average loss over {print_freq} batches: {losses.avg}")
                losses.reset()
        acc = evaluate_switch_at_layer(val_loader, model, layer, device)
        print(f"Accuracy at end of epoch {epoch}", acc)

        print("Importance switches learned posteriors:")
        print(S)
        print("Switches from most important to least important:")
        print(torch.argsort(S))
        print("max: %.4f, min: %.4f" % (torch.max(S), torch.min(S)))

        is_best = acc > best_accuracy
        best_accuracy = max(acc, best_accuracy)
        if is_best:
            print("Rank for switches from most important/largest to smallest after %s " %  str(epochs))
            print(S)
            print("max: %.4f, min: %.4f" % (torch.max(S), torch.min(S)))
            print("Best updated")
            if os.path.isfile(resume):
                print("=> overwriting checkpoint'{}'".format(resume))
                checkpoint = torch.load(resume)
                importance_switches = checkpoint['importance_switches']
            else:
                importance_switches = {key:[] for key in list(vgg.vgg16_hidden_dims)}

            importance_switches[layer] = S
            save_checkpoint({
                'layer': layer,
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_accuracy,
                'optim_state_dict': optimizer.state_dict(),
                'importance_switches': importance_switches,
        }, filename=os.path.join(file_path, f'checkpoint_imp_switch_latest.tar'))
        resume = os.path.join(file_path, f'checkpoint_imp_switch_latest.tar')

        if create_bkp:
            shutil.copy(resume, file_path_bkp + 'bkp_checkpoint_imp_switch_' + 'layer_' + layer + 'epoch_' + epoch + '.tar')


def train_importance_switches(method, switch_samps, resume, original, batch_size, workers, lr, start_layer, epochs, start_epoch, print_freq, device, save_dir, create_bkp):
    train_loader, val_loader = get_train_valid_loader('./dataset',
                                                    batch_size,
                                                    augment=True,
                                                    random_seed=0,
                                                    num_workers=workers)
    vgg16_hidden_dims_list = list(vgg.vgg16_hidden_dims)
    for i in range(vgg16_hidden_dims_list.index(start_layer), len(vgg.vgg16_hidden_dims)):
        if i > vgg16_hidden_dims_list.index(start_layer):
            start_epoch = 0
        train_one_importance_switch(
                                    method=method,
                                    train_loader=train_loader,
                                    val_loader=val_loader,
                                    layer=vgg16_hidden_dims_list[i],
                                    switch_samps=switch_samps,
                                    resume=resume,
                                    original=original,
                                    batch_size=batch_size,
                                    lr=lr,
                                    start_epoch=start_epoch,
                                    epochs=epochs,
                                    device=device,
                                    print_freq=print_freq,
                                    save_dir=save_dir,
                                    create_bkp=create_bkp
                                    )
