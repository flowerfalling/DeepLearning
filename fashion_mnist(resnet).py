# -*- coding: utf-8 -*-
# @Time    : 2022/10/21 19:12
# @Author  : 之落花--falling_flowers
# @File    : fashion_mnist(resnet).py
# @Software: PyCharm
import torch
import torch.utils.data
import torchvision
import torchvision.datasets
from torch import nn
from torch.nn import functional as f
from torchvision import transforms
from pytorchsummary import summary

import base

PATH = "pth/fashion_mnist(resnet).pth"


class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, 3, stride, 1)
        self.conv2 = nn.Conv2d(num_channels, num_channels, 3, 1, 1)
        self.conv3 = nn.Conv2d(input_channels, num_channels, 1, stride) if use_1x1conv else None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        y = f.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        x = self.conv3(x) if self.conv3 else x
        y += x
        return f.relu(y)


def resnet_block(input_channels, num_channels, num_residuals, fisrt_block=False):
    block = []
    for i in range(num_residuals):
        if not i and not fisrt_block:
            block.append(Residual(input_channels, num_channels, True, stride=2))
        else:
            block.append(Residual(num_channels, num_channels))
    return block


net = nn.Sequential(nn.Sequential(nn.Conv2d(1, 64, 7, 2, 3), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(3, 2, 1)),
                    nn.Sequential(*resnet_block(64, 64, 2, True)),
                    nn.Sequential(*resnet_block(64, 128, 2)),
                    nn.Sequential(*resnet_block(128, 256, 2)),
                    nn.Sequential(*resnet_block(256, 512, 2)),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(), nn.Linear(512, 10))


def train(epoch=1, save=False):
    try:
        net.load_state_dict(torch.load(PATH, map_location=torch.device('xpu')))
        pass
    except FileNotFoundError:
        pass
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5),
                                    transforms.Resize(224, interpolation=transforms.InterpolationMode.NEAREST)])
    trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    trainLoader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
    for e in range(epoch):
        running_loss = 0.0
        i = 0
        for data, target in trainLoader:
            outcome = net(data)
            optimizer.zero_grad()
            loss = criterion(outcome, target)
            loss.backward()
            optimizer.step()
            running_loss += loss
            i += 1
            if i % 1000 == 999:
                print(
                    'epoch: %d, complete: (%5d/60000), loss: %.5f' % ((e + 1), (i + 1) * 4, (running_loss / 1000)))
                running_loss = 0.0
        if save:
            torch.save(net.state_dict(), PATH)


def test():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5),
                                    transforms.Resize(224, interpolation=transforms.InterpolationMode.NEAREST)])
    testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    testLoader = torch.utils.data.DataLoader(testset, shuffle=True, num_workers=2)
    base.test(net, PATH, testLoader)


def info():
    summary((1, 224, 224), model=net)


@base.timer
def main():
    train(3, True)
    test()
    pass


if __name__ == '__main__':
    main()
