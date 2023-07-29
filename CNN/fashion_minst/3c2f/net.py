# -*- coding: utf-8 -*-
# @Time    : 2023/7/26 18:01
# @Author  : 之落花--falling_flowers
# @File    : net.py
# @Software: PyCharm
import torch
from torch import nn
from torch.nn import functional as f


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 4)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.drop1 = nn.Dropout(p=0.2)
        self.drop2 = nn.Dropout(p=0.5)

    def forward(self, x):
        x = f.max_pool2d(f.relu(self.conv1(x)), (2, 2))
        x = f.max_pool2d(f.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = f.relu(self.fc1(self.drop1(x)))
        x = f.relu(self.fc2(self.drop2(x)))
        x = self.fc3(x)
        return x
