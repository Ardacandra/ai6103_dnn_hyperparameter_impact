import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

class CustomBatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, alpha=0.9):
        """
        Custom class to implement batch normalization 
        where the channel-wise mean and variance does not participate in gradient calculation.

        Normalization will be done over the channel dimension "C"
        from a 4 dimensional input with size (N, C, H, W).

        Args:
            num_features (int): channel count (C) from input (N, C, H, W)
            eps (float): small constant to avoid division by zero
            alpha (float): decay factor running mean and variance update

        Shape:
            - Input : (N, C, H, W)
            - Output : (N, C, H, W)

        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.alpha = alpha

        #initialize learnable parameters
        self.weight = nn.Parameter(torch.ones(1, self.num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, self.num_features, 1, 1))

        #initialize running stats for inference
        #use register_buffer to make it persistent but not trainable
        self.register_buffer("running_mean", torch.zeros(1, num_features, 1, 1))
        self.register_buffer("running_var", torch.ones(1, num_features, 1, 1))

    def forward(self, x):
        if self.training:
            #compute the mean and variance per channel
            #detach x to prevent gradient flow
            mean = x.detach().mean(dim=(0, 2, 3), keepdim=True)
            var = x.detach().var(dim=(0, 2, 3), unbiased=False, keepdim=True) #unbiased=False -> divide by NHW, not NHW-1

            #update running statistics
            self.running_mean.mul_(self.alpha).add_((1 - self.alpha) * mean)
            self.running_var.mul_(self.alpha).add_((1 - self.alpha) * var)
        else:
            #use running stats for inference
            mean = self.running_mean
            var = self.running_var

        #apply batch normalization
        out = self.weight * (x - mean) / torch.sqrt(var + self.eps) + self.bias
        return out

# defining resnet models

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = CustomBatchNorm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = CustomBatchNorm(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                CustomBatchNorm(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = CustomBatchNorm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = CustomBatchNorm(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = CustomBatchNorm(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                CustomBatchNorm(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        # This is the "stem"
        # For CIFAR (32x32 images), it does not perform downsampling
        # It should downsample for ImageNet
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = CustomBatchNorm(64)
        # four stages with three downsampling
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def test_resnet18():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())