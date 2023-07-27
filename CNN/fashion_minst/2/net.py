# -*- coding: utf-8 -*-
# @Time    : 2023/7/26 18:12
# @Author  : 之落花--falling_flowers
# @File    : net.py
# @Software: PyCharm
from torch import nn
from torch.nn import functional as f


class ResBlock(nn.Module):
    def __init__(self, ic, oc, use_1x1=False, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(ic, oc, 3, stride, 1)
        self.conv2 = nn.Conv2d(oc, oc, 3, 1, 1)
        self.conv3 = nn.Conv2d(ic, oc, 1, stride) if use_1x1 else None
        self.bn1 = nn.BatchNorm2d(oc)
        self.bn2 = nn.BatchNorm2d(oc)

    def forward(self, x):
        y = f.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        y += self.conv3(x) if self.conv3 else x
        return f.relu(y)


class Net(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Net, self).__init__(*args, **kwargs)
        self.begin = nn.Sequential(
            nn.Conv2d(1, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1)
        )
        self.hidden = nn.Sequential(
            nn.Sequential(
                ResBlock(64, 64),
                ResBlock(64, 64)
            ),
            nn.Sequential(
                ResBlock(64, 128, True, 2),
                ResBlock(128, 128)
            ),
            nn.Sequential(
                ResBlock(128, 256, True, 2),
                ResBlock(256, 256)
            ),
            nn.Sequential(
                ResBlock(256, 512, True, 2),
                ResBlock(512, 512)
            ),
        )
        self.out = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        y = self.begin(x)
        y = self.hidden(y)
        y = self.out(y)
        return y
