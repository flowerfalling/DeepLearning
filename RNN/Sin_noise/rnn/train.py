# -*- coding: utf-8 -*-
# @Time    : 2023/8/6 17:54
# @Author  : 之落花--falling_flowers
# @File    : train.py
# @Software: PyCharm
import random
from typing import Union

import torch
from matplotlib import pyplot as plt
from rich.progress import track
from torch import nn

import base
from net import Net

EPOCH = 1000
PATH = r"D:\Projects\PycharmProjects\DeepLearning\pth\RNN\Sin-noise\rnn\1.pth"

num_steps = 49
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
loss_list = []

net = Net().to(device)
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
criterion = nn.MSELoss().to(device)
net.train()


@base.ringer()
@base.timer
def train(epo=1, load: Union[bool, str] = PATH, save: Union[bool, str] = False):
    global loss_list
    if load:
        try:
            net.load_state_dict(torch.load(load, map_location=device))
        except FileNotFoundError as e:
            print(e)
            exit(-1)
    begin_state = torch.zeros(1, 1, 32, device=device).detach_()
    for epoch in track(range(epo), description="Training..."):
        start = random.randint(0, 3)
        data = torch.sin(torch.linspace(start, start + 20, 100)) + torch.normal(0, 0.2, (100,))
        x = torch.cat([data[i:-5 + i, None] for i in range(4)], 1)[None]
        t = data[5:].reshape(1, 95, 1)
        y, _ = net(x, begin_state)
        loss = criterion(y, t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'epoch: {epoch + 1}\tloss: {loss}')
        loss_list.append(loss.to('cpu').detach())
    if save:
        torch.save(net.state_dict(), save)


def main():
    train(EPOCH, False)
    plt.plot(loss_list)
    plt.show()


if __name__ == '__main__':
    main()
