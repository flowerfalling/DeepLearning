# -*- coding: utf-8 -*-
# @Time    : 2023/8/6 15:17
# @Author  : 之落花--falling_flowers
# @File    : test.py
# @Software: PyCharm
import random

import matplotlib.pyplot as plt
import torch

from net import Net

PATH = "D:\\Projects\\PycharmProjects\\Deep-learning\\pth\\RNN\\Sin-smooth\\rnn\\1.pth"


def test(length):
    net = Net()
    net.train(False)
    try:
        net.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
    except FileNotFoundError as e:
        print(e)
        exit(-1)
    start = random.randint(0, 3)
    x = torch.linspace(start, start + length, length * 5)
    data = torch.sin(torch.linspace(start, start + length, length * 5))
    out_puts = [data[0]]
    state = torch.zeros(1, 1, 32)
    for y in data[1:length]:
        _, state = net(out_puts[-1].reshape(1, 1, 1), state)
        out_puts.append(y)
    for _ in range(length * 4):
        y, state = net(out_puts[-1].reshape(1, 1, 1), state)
        out_puts.append(y)
    plt.plot(x, data)
    plt.plot(x, torch.tensor(out_puts))
    plt.scatter(x, data, s=5)
    plt.scatter(x, torch.tensor(out_puts), s=5)
    plt.show()


def main():
    test(50)


if __name__ == '__main__':
    main()
