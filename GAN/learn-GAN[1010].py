# -*- coding: utf-8 -*-
# @Time    : 2022/12/24 11:38
# @Author  : 之落花--falling_flowers
# @File    : learn-GAN.py
# @Software: PyCharm
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas
import torch
from torch import nn
from abc import ABCMeta

PATH_G = '../pth/GAN_1010G.pth'
PATH_D = '../pth/GAN_1010D.pth'


class Plot(metaclass=ABCMeta):
    progress = []

    def plot_progress(self):
        df = pandas.DataFrame(self.progress, columns=['loss'])
        df.plot(ylim=(0, 1.0), figsize=(16, 8), alpha=0.1, marker='.', grid=True, yticks=(0, 0.25, 0.5))


class GR:
    __real_data = torch.FloatTensor([
        random.uniform(0.8, 1.0),
        random.uniform(0.0, 0.2),
        random.uniform(0.8, 1.0),
        random.uniform(0.0, 0.2),
    ])

    def __call__(self, *args, **kwargs):
        return self.__real_data


class D(nn.Module, Plot):
    def __init__(self):
        super(D, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 3),
            nn.Sigmoid(),
            nn.Linear(3, 1),
            nn.Sigmoid()
        )

        self.loss_func = nn.MSELoss()
        self.optimiser = torch.optim.SGD(self.parameters(), lr=0.01)
        self.counter = 0

    def forward(self, inputs):
        return self.model(inputs)

    def nn_train(self, inputs, targets):
        outputs = self(inputs)
        loss = self.loss_func(outputs, targets)
        self.counter += 1
        if not self.counter % 10:
            self.progress.append(loss.item())
        if not self.counter % 10000:
            print("counter = ", self.counter)
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()


class G(nn.Module, Plot):
    def __init__(self):
        super(G, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 3),
            nn.Sigmoid(),
            nn.Linear(3, 4),
            nn.Sigmoid()
        )

        self.optimiser = torch.optim.SGD(self.parameters(), lr=0.01)
        self.counter = 0
        self.image_list = []

    def forward(self, inputs):
        return self.model(inputs)

    def nn_train(self, d: D, inputs, targets):
        g_outputs = self(inputs)
        d_outputs = d(g_outputs)
        loss = d.loss_func(d_outputs, targets)
        self.counter += 1
        if not self.counter % 10:
            self.progress.append(loss.item())
        if not self.counter % 1000:
            self.image_list.append(g_outputs.detach().numpy())
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()


def main():
    discriminator = D()
    generator = G()
    try:
        generator.load_state_dict(torch.load(PATH_G))
        discriminator.load_state_dict(torch.load(PATH_D))
    except FileNotFoundError:
        pass
    for _ in range(10000):
        discriminator.nn_train(GR()(), torch.FloatTensor([1.0]))
        discriminator.nn_train(generator(torch.FloatTensor([0.5])).detach(), torch.FloatTensor([0.0]))
        generator.nn_train(discriminator, torch.FloatTensor([0.5]), torch.FloatTensor([1.0]))
    discriminator.plot_progress()
    plt.figure(figsize=(16, 8))
    plt.imshow(np.array(generator.image_list).T, interpolation='none', cmap='Blues')
    plt.show()
    torch.save(generator.state_dict(), PATH_G)
    torch.save(discriminator.state_dict(), PATH_D)
    pass


if __name__ == '__main__':
    main()
