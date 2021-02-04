"""
BagNet that takes as input the order1 scattering transform of an image.
The receptive fiels is not clearly defined since:
    - the filters used in scattering transform have Gaussian windows that have theoritically infinite support
    - two convolutions are done for the order1 coefficients
"""
from __future__ import division
import torch.nn as nn
import torch

class BagNetScattering(nn.Module):
    """BagNet Using scattering.
    """

    def __init__(self, J, N, layer_width=1024, num_classes=1000, order2=False,
                 n_iterations=4, first_layer_kernel_size=1, skip_stride=1):
        super(BagNetScattering, self).__init__()

        self.nfscat = 3 * (1 + 8*J)
        if order2:
            self.nfscat += 3 * 8**2 * J*(J - 1)// 2

        self.bn0 = nn.BatchNorm2d(self.nfscat)
        self.conv1 = nn.Conv2d(self.nfscat, layer_width, kernel_size=first_layer_kernel_size, stride=skip_stride, padding=0, bias=True)


        modules = []
        for _ in range(n_iterations-1):
            modules.append(nn.BatchNorm2d(layer_width))
            modules.append(nn.Conv2d(layer_width, layer_width, kernel_size=1, stride=1, padding=0, bias=True))
            modules.append(nn.ReLU())
        self.layers = nn.Sequential(*modules)


        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.bn = nn.BatchNorm2d(layer_width)
        self.fc = nn.Linear(layer_width, num_classes)


    def forward(self, x):
        x = x.view(x.size(0), x.size(1)*x.size(2), x.size(3), x.size(4))
        x = self.bn0(x)
        x = self.conv1(x)
        x = self.layers(x)
        x = self.bn(x)
        x = self.avgpool(x)
        output = x.view(x.size(0), -1)
        output = self.fc(output)
        return output


class ScatteringLinear(nn.Module):
    """BagNet Using scattering.
    """

    def __init__(self, n_space, J, num_classes=1000, order2=False, use_fft=False):
        super(ScatteringLinear, self).__init__()

        self.nfscat = 3 * (1 + 8*J)
        if order2:
            self.nfscat += 3 * 8**2 * J*(J - 1)// 2
        self.use_fft = use_fft
        if use_fft:
            self.nfscat *= 3
        self.avg_pool = nn.AvgPool2d(3, 2)
        self.n_space = self.avg_pool(torch.zeros(1,1,n_space,n_space)).size(2)

        self.bn0 = nn.BatchNorm2d(self.nfscat)
        self.fc = nn.Linear(self.nfscat*self.n_space**2, num_classes)


    def forward(self, x):
        if self.use_fft:
            with torch.no_grad():
                x_ = x.reshape(x.size() + (1,))
                x_ = torch.cat([x_, torch.zeros_like(x_)], dim=-1)
                x_color = x_.transpose(1, 4)
                x_fft_color = torch.sqrt((torch.fft(x_color, 1)**2).sum(dim=-1)).transpose(1,4)
                x_channel = x_.transpose(2, 4)
                x_fft_channel = torch.sqrt((torch.fft(x_channel, 1)**2).sum(dim=-1)).transpose(2,4)
                x = torch.cat([x, x_fft_color, x_fft_channel], dim=2)
        x = x.view(x.size(0), x.size(1)*x.size(2), x.size(3), x.size(4))
        x = self.bn0(x)
        x = self.avg_pool(x)
        output = x.view(x.size(0), -1)
        output = self.fc(output)
        return output


