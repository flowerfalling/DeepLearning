# -*- coding: utf-8 -*-
# @Time    : 2023/7/25 20:46
# @Author  : 之落花--falling_flowers
# @File    : train.py
# @Software: PyCharm
from typing import Union

import matplotlib.pyplot as plt
import torch
from rich.progress import track
from torch import nn

import base
from CNN.minst.data import loader
from net import Net

EPOCH = 20
PATH = r"D:\Projects\PycharmProjects\DeepLearning\pth\CNN\mnist\1\1.pth"

batch_size = 32
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
loss_list = []

trainloader = loader(True, batch_size)
net = Net().to(device)
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss().to(device)
dataset_length = len(trainloader)
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
    for epoch in track(range(epo), description='Training...'):
        epo_loss = 0
        for batch in trainloader:
            data, target = [i.to(device) for i in batch]
            optimizer.zero_grad()
            outcome = net(data)
            loss = criterion(outcome, target)
            epo_loss += loss.to('cpu').detach()
            loss.backward()
            optimizer.step()
        epo_loss /= (dataset_length / batch_size).__ceil__()
        print(f'epoch: {epoch + 1}\tloss: {epo_loss}')
        loss_list.append(epo_loss)
    if save:
        torch.save(net.state_dict(), save)


def main():
    train(EPOCH, False)
    plt.plot(loss_list)
    plt.show()


if __name__ == '__main__':
    main()
