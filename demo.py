# -*- coding: utf-8 -*-
# @Time    : 2022/10/4 21:52
# @Author  : 之落花--falling_flowers
# @File    : demo.py
# @Software: PyCharm
import torch
import torch.utils.data
import torchvision
from torch import nn
from torchvision import transforms

import base


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(5, 1, 4, 4),
            # nn.ConvTranspose2d(6, 1, 3, 2)
        )

    def forward(self, x):
        x = self.main(x)
        return x


def train():
    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
    # trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
    net = Net()
    demo = torch.randn(5, 7, 7)
    out = net(demo)
    pass
    # for (img, _) in trainloader:
    #     out = net(img)
    #     pass


@base.timer
def main():
    train()
    pass


if __name__ == '__main__':
    main()
    pass
