import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import vgg
from dataloader import get_train_valid_loader
from util import AverageMeter, save_checkpoint, create_dir_if_not_exists
from evaluate import evaluate, evaluate_switch_at_layer, accuracy


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


def train_one_importance_switch(method, train_loader, val_loader, lr, epochs, layer, switch_samps, device, resume, batch_size, workers, print_freq, save_dir):

    file_path = os.path.join(save_dir, 'models', method)
    create_dir_if_not_exists(file_path)

    model = vgg.vgg16_bn(method=method, switch_samps=switch_samps, layer=layer, device=device).to(device)
    # criterion = nn.CrossEntropyLoss()

    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(eval, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(resume))

    if method == "dirichlet":
        optimizer = optim.Adam([model.switch_parameter_alpha], lr=lr)
    elif method == "generalized_dirichlet":
        pass

    model.train()
    stop = 0
    epoch = 0
    best_accuracy = 0
    entry = np.zeros(3)
    best_model = -1
    how_many_epochs = 200
    annealing_steps = float(8000. * how_many_epochs)
    beta_func = lambda s: min(s, annealing_steps) / annealing_steps

    for epochs in range(epochs):
        epoch=epoch+1
        annealing_rate = beta_func(epoch)
        model.train()
        evaluate_switch_at_layer(model, layer)
        for i, data in enumerate(train_loader):
            inputs, labels=data
            inputs, labels=inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs, S = model(inputs, layer)
            alpha_0 = 2
            loss = loss_function_dirichlet(outputs, labels, S, alpha_0, vgg.vgg16_hidden_dims[layer], batch_size, annealing_rate)
            loss.backward()
            #print(net2.c1.weight.grad[1, :])
            #print(net2.c1.weight[1, :])
            optimizer.step()
            if i % print_freq == 0:
               print (i)
               print (loss.item())
            #    evaluate()
        #print (i)
        print (loss.item())
        accuracy = evaluate_switch_at_layer(model, layer)
        print ("Epoch " +str(epoch)+ " ended.")

        print("S")
        print(S)
        print(torch.argsort(S))
        print("max: %.4f, min: %.4f" % (torch.max(S), torch.min(S)))

        if (accuracy<=best_accuracy):
            stop=stop+1
            entry[2]=0
        else:
            best_accuracy=accuracy
            print("Best updated")
            stop=0
            entry[2]=1
            best_model=model.state_dict()
            best_optim=optimizer.state_dict()
            save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_accuracy,
            'optim_state_dict': best_optim
        }, filename=os.path.join(file_path, f'checkpoint_imp_switch_layer_{layer}_epoch{epoch}.tar'))
            # torch.save({'model_state_dict' : best_model, 'optimizer_state_dict': best_optim}, "models/%s_conv:%d_conv:%d_fc:%d_fc:%d_rel_bn_drop_trainval_modelopt%.1f_epo:%d_acc:%.2f" % (dataset, conv1, conv2, fc1, fc2, trainval_perc, epoch, best_accuracy))

        print("\n")
        #write
        # entry[0]=accuracy; entry[1]=loss
        # with open(filename, "a+") as file:
        #     file.write(",".join(map(str, entry))+"\n")
    return best_accuracy, epoch, best_model, S


def train_importance_switches(method, switch_samps, resume, batch_size, workers, lr, epochs, print_freq, device, save_dir):
    file_path = os.path.join(save_dir, 'importance_switches', method)
    create_dir_if_not_exists(file_path)
    train_loader, val_loader = get_train_valid_loader('./dataset',
                                                    batch_size,
                                                    augment=True,
                                                    random_seed=0,
                                                    num_workers=workers)
    switch_data={}
    switch_data['combinationss'] = []
    switch_data['switches'] = []
    for layer in vgg.vgg16_hidden_dims.keys():
        best_accuracy, epoch, best_model, S = train_one_importance_switch(
                                                                        method=method,
                                                                        train_loader=train_loader,
                                                                        val_loader=val_loader,
                                                                        layer=layer,
                                                                        switch_samps=switch_samps,
                                                                        resume=resume,
                                                                        batch_size=batch_size,
                                                                        lr=lr,
                                                                        epochs=epochs,
                                                                        device=device,
                                                                        workers=workers,
                                                                        print_freq=print_freq,
                                                                        save_dir=save_dir
                                                                        )
        print("Rank for switches from most important/largest to smallest after %s " %  str(epochs))
        print(S)
        print("max: %.4f, min: %.4f" % (torch.max(S), torch.min(S)))
        ranks_sorted = np.argsort(S.cpu().detach().numpy())[::-1]
        print(",".join(map(str, ranks_sorted)))
        switch_data['combinationss'].append(ranks_sorted); switch_data['switches'].append(S.cpu().detach().numpy())
    print('*'*30)
    print(switch_data['combinationss'])
    # combinationss=switch_data['combinationss']
    file_path=os.path.join(file_path, f"switch_samps_{switch_samps}_epochs_{epochs}")
    np.save(file_path, switch_data)