# -*- coding: utf-8 -*-
# @Time    : 2023/8/2 18:54
# @Author  : 之落花--falling_flowers
# @File    : train.py
# @Software: PyCharm
import sys

import torch
from torch import nn
from rich.progress import track

sys.path.append('..')
sys.path.append('../../..')

from net import Net
import data
import base

PATH = "D:\\Projects\\PycharmProjects\\Deep-learning\\pth\\RNN\\The Time Machine\\RNN\\1.pth"
EPOCH = 100

batch_size, num_steps = 32, 35


def train(epo=1, load=True, save=False, save_path=PATH):
    net: nn.Module = Net()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.05, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    if load:
        try:
            net.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
        except FileNotFoundError as e:
            print(e)
            exit(-1)
    train_iter, vocab = data.load_data_time_machine(batch_size, num_steps)
    for epoch in track(range(epo), description='Training...'):
        epo_loss = 0
        for x, t in train_iter:
            state = Net.begin_state(batch_size)
            state.detach_()
            y, state = net(x, state)
            optimizer.zero_grad()
            loss = criterion(y.reshape(-1, y.shape[-1]), t.reshape(-1))
            epo_loss += loss
            loss.backward()
            optimizer.step()
        print(f'epoch: {epoch + 1}\tloss: {epo_loss / 8}')
    if save:
        torch.save(net.state_dict(), save_path)


@base.ringer
@base.timer
def main():
    train(EPOCH, True, True)


if __name__ == '__main__':
    main()
