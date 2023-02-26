#https://github.com/chengyangfu/pytorch-vgg-cifar10/blob/master/vgg.py

'''
Modified from https://github.com/pytorch/vision.git
'''
import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.distributions import Gamma
import torch.nn.functional as F
import collections
from util import Method

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


vgg16_split = {
    "conv1": 1,
    "conv2": 4,
    "conv3": 8,
    "conv4": 11,
    "conv5": 15,
    "conv6": 18,
    "conv7": 21,
    "conv8": 25,
    "conv9": 28,
    "conv10": 31,
    "conv11": 35,
    "conv12": 38,
    "conv13": 41
}

vgg16_hidden_dims = {
    "conv1": 64,
    "conv2": 64,
    "conv3": 128,
    "conv4": 128,
    "conv5": 256,
    "conv6": 256,
    "conv7": 256,
    "conv8": 512,
    "conv9": 512,
    "conv10": 512,
    "conv11": 512,
    "conv12": 512,
    "conv12": 512,
    "fc1": 512,
    "fc2": 512
}


class VGG(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, features, method=None, switch_samps=None, hidden_dim=None, device=torch.device('cuda')):
        super(VGG, self).__init__()
        self.features = features
        self.fc_1 = nn.Linear(512, 512)
        self.fc_2 = nn.Linear(512, 512)
        self.fc_3 = nn.Linear(512, 10)
        self.dropout = nn.Dropout()

        # self.classifier = nn.Sequential(
        #     nn.Dropout(),
        #     nn.Linear(512, 512),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(512, 512),
        #     nn.ReLU(True),
        #     nn.Linear(512, 10),
        # )
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

        self.method = method
        self.switch_samps = switch_samps
        self.device = device

        if method == Method.DIRICHLET:
            self.switch_parameter_alpha = Parameter(-1*torch.ones(vgg16_hidden_dims[hidden_dim]), requires_grad=True)
        elif method == Method.GENERALIZED_DIRICHLET:
            pass

    def switch_multiplication(self, output, SstackT):
        rep = SstackT.unsqueeze(2).unsqueeze(2).repeat(1, 1, output.shape[2], output.shape[3])  # (150,10,24,24)
        # output is (100,10,24,24), we want to have 100,150,10,24,24, I guess
        output = torch.einsum('ijkl, mjkl -> imjkl', (rep, output))
        output = output.view(output.shape[0] * output.shape[1], output.shape[2], output.shape[3], output.shape[4])
        return output, SstackT


    def switch_multiplication_fc(self, output, SstackT):
        output = torch.einsum('ij, mj -> imj', (SstackT, output))
        output = output.reshape(output.shape[0] * output.shape[1], output.shape[2])
        return output, SstackT


    def forward(self, x, switch_layer=None):
        BATCH_SIZE = x.shape[0]
        if self.method == Method.DIRICHLET:
            phi = F.softplus(self.switch_parameter_alpha)
            """ draw Gamma RVs using phi and 1 """
            num_samps = self.switch_samps
            concentration_param = phi.view(-1, 1).repeat(1, num_samps).to(self.device)
            beta_param = torch.ones(concentration_param.size()).to(self.device)
            # Gamma has two parameters, concentration and beta, all of them are copied to 200,150 matrix
            Gamma_obj = Gamma(concentration_param, beta_param)
            gamma_samps = Gamma_obj.rsample()  # 200, 150, hidden_dim x samples_num

            if any(torch.sum(gamma_samps, 0) == 0):
                print("sum of gamma samps are zero!")
            else:
                Sstack = gamma_samps / torch.sum(gamma_samps, 0)  # 1dim - number of neurons (200), 2dim - samples (150)

            SstackT = Sstack.t()

        elif self.method == Method.GENERALIZED_DIRICHLET:
            pass



        if switch_layer is not None:
            if switch_layer in vgg16_split:
                x = self.features[:vgg16_split[switch_layer]](x)
                x, SstackT_ret=self.switch_multiplication(x, SstackT)
                x = self.features[vgg16_split[switch_layer]:](x)
            x = x.view(x.size(0), -1)
            x = self.fc_1(x)
            if switch_layer == "fc_1":
                x, SstackT_ret = self.switch_multiplication_fc(x, SstackT)
            x = self.fc_2(x)
            if switch_layer == "fc_2":
                x, SstackT_ret = self.switch_multiplication_fc(x, SstackT)
            x = self.fc_3(x)
            x = x.reshape(BATCH_SIZE, self.switch_samps, -1)
            x = torch.mean(x, 1)

        else:
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.fc_1(x)
            x = self.fc_2(x)
            x = self.fc_3(x)

        if self.method == Method.DIRICHLET:
            return x, phi
        elif self.method == Method.GENERALIZED_DIRICHLET:
            pass
        else:
            return x


def make_layers(cfg, batch_norm=False):
    layers = collections.OrderedDict()
    in_channels = 3
    for i, v in enumerate(cfg):
        if v == 'M':
            layers[f"MaxPool2d{i}"] = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers[f"Conv2d_{i}"] = conv2d
                layers[f"BatchNorm2d_{i}"] = nn.BatchNorm2d(v)
                layers[f"ReLU_{i}"] = nn.ReLU(inplace=True)
            else:
                layers[f"Conv2d_{i}"] = conv2d
                layers[f"ReLU_{i}"] = nn.ReLU(inplace=True)
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
          512, 512, 512, 512, 'M'],
}


# def vgg11():
#     """VGG 11-layer model (configuration "A")"""
#     return VGG(make_layers(cfg['A']))


# def vgg11_bn():
#     """VGG 11-layer model (configuration "A") with batch normalization"""
#     return VGG(make_layers(cfg['A'], batch_norm=True))


# def vgg13():
#     """VGG 13-layer model (configuration "B")"""
#     return VGG(make_layers(cfg['B']))


# def vgg13_bn():
#     """VGG 13-layer model (configuration "B") with batch normalization"""
#     return VGG(make_layers(cfg['B'], batch_norm=True))


# def vgg16():
#     """VGG 16-layer model (configuration "D")"""
#     return VGG(make_layers(cfg['D']))


def vgg16_bn(method=None, switch_samps=None, hidden_dim=None, device=torch.device('cuda')):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(make_layers(cfg['D'], batch_norm=True), method=method, switch_samps=switch_samps, hidden_dim=hidden_dim, device=device)


# def vgg19():
#     """VGG 19-layer model (configuration "E")"""
#     return VGG(make_layers(cfg['E']))


# def vgg19_bn():
#     """VGG 19-layer model (configuration 'E') with batch normalization"""
#     return VGG(make_layers(cfg['E'], batch_norm=True))
