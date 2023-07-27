# -*- coding: utf-8 -*-
# @Time    : 2023/7/25 20:46
# @Author  : 之落花--falling_flowers
# @File    : train.py
# @Software: PyCharm
import sys

import torch
import torch.utils.data
from torch import nn

sys.path.append('..')
sys.path.append('../../..')

import base
from net import Net
from data import loader

PATH = "D:\\Projects\\PycharmProjects\\Deep-learning\\pth\\CNN\\mnist\\1\\1.pth"
EPOCHS = 1


def train(epo=1, load=True, save=False, save_path=PATH):
    net = Net()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    if load:
        try:
            net.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
        except FileNotFoundError as e:
            print(e)
            exit(-1)
    trainloader = loader(True, 25)

    for epoch in range(epo):
        running_loss = 0.0
        for i, (data, target) in enumerate(trainloader):
            outcome = net(data)
            loss = criterion(outcome, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss
            if i % 1000 == 999:
                print(
                    'epoch: %d, complete: (%5d/%5d), loss: %.5f' % (epoch + 1, i + 1, len(trainloader), running_loss / 1000))
                running_loss = 0.0
        break
    if save:
        torch.save(net.state_dict(), save_path)


@base.ringer
@base.timer
def main():
    train(EPOCHS)


if __name__ == '__main__':
    main()
