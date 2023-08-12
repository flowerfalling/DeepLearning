# -*- coding: utf-8 -*-
# @Time    : 2023/8/2 18:54
# @Author  : 之落花--falling_flowers
# @File    : train.py
# @Software: PyCharm
import sys
from typing import Union

import torch
from matplotlib import pyplot as plt
from rich.progress import track
from torch import nn

from net import Net

sys.path.append('..')
sys.path.append('../../..')

import data
import base

EPOCH = 100
PATH = "D:\\Projects\\PycharmProjects\\Deep-learning\\pth\\RNN\\The Time Machine\\gru\\1.pth"

batch_size, num_steps = 32, 35
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
loss_list = []

train_iter, vocab = data.load_data_time_machine(batch_size, num_steps)
net = Net().to(device)
optimizer = torch.optim.SGD(net.parameters(), lr=0.05, momentum=0.9)
criterion = nn.CrossEntropyLoss().to(device)
net.train()


@base.ringer
@base.timer
def train(epo=1, load: Union[bool, str] = PATH, save: Union[bool, str] = False):
    global loss_list
    if load:
        try:
            net.load_state_dict(torch.load(load, map_location=device))
        except FileNotFoundError as e:
            print(e)
            exit(-1)
    for epoch in track(range(epo), description='Training...'):
        epo_loss = 0
        for batch in train_iter:
            x, t = [i.to(device) for i in batch]
            state = Net.begin_state(batch_size).to(device)
            state.detach_()
            y, _ = net(x, state)
            optimizer.zero_grad()
            loss = criterion(y.reshape(-1, y.shape[-1]), t.reshape(-1))
            epo_loss += loss.to('cpu').detach()
            loss.backward()
            optimizer.step()
        print(f'epoch: {epoch + 1}\tloss: {epo_loss / 8}')
        loss_list.append(epo_loss / 8)
    if save:
        torch.save(net.state_dict(), save)


def main():
    train(EPOCH, False)
    plt.plot(loss_list)
    plt.show()


if __name__ == '__main__':
    main()
