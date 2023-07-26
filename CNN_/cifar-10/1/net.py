# -*- coding: utf-8 -*-
# @Time    : 2023/7/26 16:17
# @Author  : 之落花--falling_flowers
# @File    : net.py
# @Software: PyCharm
import torch
from torch import nn
from torch.nn import functional as f


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 3)
        self.conv2 = nn.Conv2d(20, 100, 4)
        self.conv3 = nn.Conv2d(100, 500, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(2000, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 500)
        self.fc4 = nn.Linear(500, 200)
        self.fc5 = nn.Linear(200, 100)
        self.fc6 = nn.Linear(100, 10)

    def forward(self, x):
        y = self.pool(f.relu(self.conv1(x)))
        y = self.pool(f.relu(self.conv2(y)))
        y = self.pool(f.relu(self.conv3(y)))
        y = torch.flatten(y, 1)
        y = f.relu(self.fc1(y))
        y = f.relu(self.fc2(y))
        y = f.relu(self.fc3(y))
        y = f.relu(self.fc4(y))
        y = f.relu(self.fc5(y))
        y = self.fc6(y)
        return y
