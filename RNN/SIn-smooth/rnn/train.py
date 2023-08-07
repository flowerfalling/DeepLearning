# -*- coding: utf-8 -*-
# @Time    : 2023/8/6 15:17
# @Author  : 之落花--falling_flowers
# @File    : train.py
# @Software: PyCharm
import random
import sys

import torch
from rich.progress import track
from torch import nn

sys.path.append('../../..')

import base
from net import Net

PATH = "D:\\Projects\\PycharmProjects\\Deep-learning\\pth\\RNN\\Sin-smooth\\rnn\\1.pth"
EPOCH = 2000

num_steps = 49


def train(epo=1, load=True, save=False, save_path=PATH):
    net: nn.Module = Net()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.MSELoss()
    if load:
        try:
            net.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
        except FileNotFoundError as e:
            print(e)
            exit(-1)
    begin_state = torch.zeros(1, 1, 32).detach_()
    for epoch in track(range(epo), description="Training..."):
        start = random.randint(0, 3)
        data = torch.sin(torch.linspace(start, start + 10, 50))
        x = data[:-1].reshape(1, 49, 1)
        t = data[1:].reshape(1, 49, 1)
        y, _ = net(x, begin_state)
        loss = criterion(y, t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'epoch: {epoch + 1}\tloss: {loss}')
    if save:
        torch.save(net.state_dict(), save_path)


@base.ringer
@base.timer
def main():
    train(EPOCH, True, True)


if __name__ == '__main__':
    main()
