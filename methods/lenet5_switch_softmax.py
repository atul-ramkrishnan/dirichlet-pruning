"""
Why use a Dirichlet for the importance switches? The random sampling makes it difficult, if not impossible, to train the parameters.
Ths file implements using scalars as importance weights instead of a Dirichlet. The scalars are passed through a softmax so they sum up to 1.
Should also be faster to train.
"""

import torch.utils.data

import torch
from torch import nn, optim
import torch.nn.functional as f

import torch.utils.data
from torchvision import datasets, transforms

import numpy as np
import csv
import pdb
import os

from torch.nn.parameter import Parameter
from torch.distributions import Gamma, Beta
from importlib.machinery import SourceFileLoader

dataset_mnist = SourceFileLoader("module_mnist", "../dataloaders/dataset_mnist.py").load_module()
dataset_fashionmnist = SourceFileLoader("module_fashionmnist", "../dataloaders/dataset_fashionmnist.py").load_module()
model_lenet5 = SourceFileLoader("module_lenet", "../models/lenet5.py").load_module()
from module_fashionmnist import load_fashionmnist
from module_mnist import load_mnist
from module_lenet import Lenet


device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device="cpu"


#############################
# PARAMS

sum_average=0; conv1=10; conv2=20; fc1=100; fc2=25
hidden_dims={'c1': conv1, 'c3': conv2, 'c5': fc1, 'f6' : fc2}
verbose = False                  # Print additional info
###################################################
# DATA
BATCH_SIZE = 100
dataset = "fashionmnist"
trainval_perc = 1
train_loader, test_loader, val_loader = load_fashionmnist(BATCH_SIZE, 1.0)

##############################################################################
# NETWORK (conv-conv-fc-fc)
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


class Lenet(nn.Module):
    def __init__(self, nodesNum1, nodesNum2, nodesFc1, nodesFc2, layer, switch_init):
        super(Lenet, self).__init__()

        self.nodesNum2=nodesNum2

        self.c1=nn.Conv2d(1, nodesNum1, 5)
        self.s2=nn.MaxPool2d(2)
        self.bn1=nn.BatchNorm2d(nodesNum1)


        self.c3=nn.Conv2d(nodesNum1,nodesNum2,5)
        self.s4=nn.MaxPool2d(2)
        self.bn2=nn.BatchNorm2d(nodesNum2)
        self.c5=nn.Linear(nodesNum2*4*4, nodesFc1)
        self.f6=nn.Linear(nodesFc1,nodesFc2)
        self.f7=nn.Linear(nodesFc2,10)

        self.drop_layer = nn.Dropout(p=0.5)
        self.parameter_switch = Parameter(switch_init*torch.ones(hidden_dims[layer]),requires_grad=True)
    
    def switch_func_softmax(self, inputs, switch):
        rep = switch.unsqueeze(1).unsqueeze(1).repeat(1, inputs.shape[2], inputs.shape[3])
        output = torch.mul(rep, inputs)
        return output
        
    def switch_func_fc_softmax(self, inputs, switch):
        output = torch.mul(inputs, switch)
        return output
    

    def forward(self, x, layer):
        switch = f.softmax(self.parameter_switch, dim=0)
        output=self.c1(x)
        if layer == 'c1':
             output = self.switch_func_softmax(output, switch)

        output=f.relu(self.s2(output))
        output=self.bn1(output)
        output=self.drop_layer(output)
        output=self.c3(output)

        if layer == 'c3':
            output = self.switch_func_softmax(output, switch)

        output=f.relu(self.s4(output))
        output=self.bn2(output)
        output=self.drop_layer(output)
        output=output.view(-1, self.nodesNum2*4*4)

        output=self.c5(output)

        if layer == 'c5':
            output = self.switch_func_fc_softmax(output, switch)

        output=self.f6(output)

        if layer == 'f6':
            output = self.switch_func_fc_softmax(output, switch)

        output = self.f7(output)
        return output, switch


nodesNum1, nodesNum2, nodesFc1, nodesFc2=10,20,100,25

# EVALUATE

def evaluate(net2, layer):
    # print('Prediction when network is forced to predict')
    net2.eval()
    correct = 0
    total = 0
    for j, data in enumerate(test_loader):
        images, labels = data
        images = images.to(device)
        #dummy works as it should, if we don't execute switch function in forward the accuracy should be original, 99.27
        #predicted_prob = net2.forward(images, "dummy")[0]  # if using switches
        #predicted_prob = net2.forward(images, "c1")[0] #13.68 for 99.27
        #predicted_prob = net2.forward(images, "c3")[0] #11.35
        predicted_prob = net2.forward(images, layer)[0]
        predicted = np.argmax(predicted_prob.cpu().detach().numpy(), axis=1)
        total += labels.size(0)
        correct += (predicted == labels.numpy()).sum().item()
    accuracy = 100 * float(correct) / total
    print("accuracy: %.2f %%" % (accuracy))
    return accuracy

###################################################
# RUN TRAINING

def run_experiment(epochs_num, layer, nodesNum1, nodesNum2, nodesFc1, nodesFc2, switch_init, path):
    print("\nRunning experiment\n")

    #CHECK WHY THSI CHANGES SO MUCH
    net2 = Lenet(nodesNum1, nodesNum2, nodesFc1, nodesFc2, layer, switch_init).to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam([net2.parameter_switch], lr=0.001)

    print(path)
    net2.load_state_dict(torch.load(path)['model_state_dict'], strict=False)


    print("Evaluate:\n")
    evaluate(net2, layer)

    # Freezing the weights of the network while we're training the importance switches
    net2.c1.weight.requires_grad = False
    net2.c3.weight.requires_grad = False
    net2.c5.weight.requires_grad = False
    net2.f6.weight.requires_grad = False
    net2.c1.bias.requires_grad = False
    net2.c3.bias.requires_grad = False
    net2.c5.bias.requires_grad = False
    net2.f6.bias.requires_grad = False
    net2.bn1.weight.requires_grad = False
    net2.bn1.bias.requires_grad = False
    net2.bn2.weight.requires_grad = False
    net2.bn2.bias.requires_grad = False
    net2.f7.weight.requires_grad = False
    net2.f7.bias.requires_grad = False



    # h = net2.c1.weight.register_hook(lambda grad: grad * 0)
    # h = net2.c3.weight.register_hook(lambda grad: grad * 0)
    # h = net2.c5.weight.register_hook(lambda grad: grad * 0)
    # h = net2.f6.weight.register_hook(lambda grad: grad * 0)
    # h = net2.c1.bias.register_hook(lambda grad: grad * 0)
    # h = net2.c3.bias.register_hook(lambda grad: grad * 0)
    # h = net2.c5.bias.register_hook(lambda grad: grad * 0)
    # h = net2.f6.bias.register_hook(lambda grad: grad * 0)
    # h = net2.bn1.weight.register_hook(lambda grad: grad * 0)
    # h = net2.bn1.bias.register_hook(lambda grad: grad * 0)
    # h = net2.bn2.weight.register_hook(lambda grad: grad * 0)
    # h = net2.bn2.bias.register_hook(lambda grad: grad * 0)
    # h = net2.f7.weight.register_hook(lambda grad: grad * 0)
    # h = net2.f7.bias.register_hook(lambda grad: grad * 0)

    #print("Retraining\n")
    net2.train()
    stop=0; epoch=0; best_accuracy=0; entry=np.zeros(3); best_model=-1
    # while (stop<early_stopping):
    for epochs in range(epochs_num):
        losses = AverageMeter()
        epoch=epoch+1
        print("Epoch: ", epoch)
        net2.train()
        for i, data in enumerate(train_loader):
            inputs, labels=data
            inputs, labels=inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs, switches = net2(inputs, layer)

            hidden_dim = hidden_dims[layer]
            loss = f.cross_entropy(outputs, labels)
            loss.backward()
            losses.update(loss.item(), inputs.size(0))
            optimizer.step()
        print ("Loss: ", losses.avg)
        evaluate(net2, layer)
        print ("Epoch " +str(epoch)+ " ended.")
        accuracy = evaluate(net2, layer)
        print("S")
        print(switches)
        print(torch.argsort(switches))
        print("max: %.4f, min: %.4f" % (torch.max(switches), torch.min(switches)))

        if (accuracy<=best_accuracy):
            stop=stop+1
            entry[2]=0
        else:
            best_accuracy=accuracy
            print("Best updated")
            stop=0
            entry[2]=1
            best_model=net2.state_dict()
            best_optim=optimizer.state_dict()
            # print(os.getcwd())
            # Hardcoded filepath for now. Fix later
            torch.save({'model_state_dict' : best_model, 'optimizer_state_dict': best_optim}, "./checkpoint/prune_and_retrain/fashionmnist/softmax/prune/%s_conv:%d_conv:%d_fc:%d_fc:%d_rel_bn_drop_trainval_modelopt%.1f_epo:%d_acc:%.2f" % (dataset, conv1, conv2, fc1, fc2, trainval_perc, epoch, best_accuracy))

        print("\n")
    return best_accuracy, epoch, best_model, switches




########################################################
# PARAMS
# epochs_num=10
# sum_average=0; conv1=10; conv2=20; fc1=100; fc2=25
# dataset = "fashionmnist"
# trainval_perc = 1
# filename="%s_test_conv_relu_bn_drop_trainval%.1f_conv:%d_conv:%d_fc:%d_fc:%d.txt" % (dataset, trainval_perc, conv1, conv2, fc1, fc2)
# filename="%s_test_conv_relu_bn_drop_trainval%.1f_conv:%d_conv:%d_fc:%d_fc:%d.txt" % (dataset, trainval_perc, conv1, conv2, fc1, fc2)


######################################################
#single run  avergaed pver n iterations  

# if __name__=='__main__':
#     for i in range(1):
#         with open(filename, "a+") as file:
#             file.write("\nInteration: "+ str(i)+"\n")
#             print("\nIteration: "+str(i))
#         best_accuracy, num_epochs, best_model=run_experiment(epochs_num, layer, conv1, conv2, fc1, fc2, num_samps_for_switch, path_full)
#         sum_average+=best_accuracy
#         average_accuracy=sum_average/(i+1)

#         with open(filename, "a+") as file:
#             file.write("\nAv accuracy: %.2f, best accuracy: %.2f\n" % (average_accuracy, best_accuracy))
#         print("\nAv accuracy: %.2f, best accuracy: %.2f\n" % (average_accuracy, best_accuracy))
#         #torch.save(best_model, filename_model)

#     #multiple runs

#     # for i1 in range(1,20):
#     #     for i2 in range(1,20):
#     #         with open(filename, "a+") as file:
#     #             file.write("\n\nnumber of hidden nodes 1: "+str(i1)+", hidden nodes 2: " +str(i2)+"\n")
#     #             print("\n\nnumber of hidden nodes 1: "+str(i1)+", hidden nodes 2: " +str(i2)+"\n")

#     #         best_accuracy, num_epochs=run_experiment(i1, i2)
#     #         with open(filename, "a+") as file:
#     #             file.write("Best accuracy: %.2f in %d epochs" % (best_accuracy, num_epochs-early_stopping))
#     #             print("Best accuracy: %.2f in %d epochs" % (best_accuracy, num_epochs-early_stopping))