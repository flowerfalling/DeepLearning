# -*- coding: utf-8 -*-
# @Time    : 2023/8/6 17:54
# @Author  : 之落花--falling_flowers
# @File    : test.py
# @Software: PyCharm
import random

import matplotlib.pyplot as plt
import torch

from net import Net

PATH = "D:\\Projects\\PycharmProjects\\Deep-learning\\pth\\RNN\\Sin-noise\\rnn\\1.pth"
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

net = Net().to(device)
net.eval()

try:
    net.load_state_dict(torch.load(PATH, map_location=device))
except FileNotFoundError as e:
    print(e)
    exit(-1)


def test(length):
    start = random.randint(0, 3)
    x = torch.linspace(start, start + length, length * 5)
    data = torch.sin(torch.linspace(start, start + length, length * 5, device=device)) + torch.normal(0, 0.2, (length * 5,))
    out_puts = [*data[:4]]
    state = torch.zeros(1, 1, 32, device=device)
    for y in data[4:length]:
        _, state = net(torch.tensor(out_puts[-4:]).reshape(1, 1, 4), state)
        out_puts.append(y)
    for _ in range(length * 4):
        y, state = net(torch.tensor(out_puts[-4:]).reshape(1, 1, 4), state)
        out_puts.append(y)
    plt.plot(x, data.to('cpu'))
    plt.plot(x, torch.tensor(out_puts, device=torch.device('cpu')))
    plt.scatter(x, data.to('cpu'), s=5)
    plt.scatter(x, torch.tensor(out_puts, device=torch.device('cpu')), s=5)
    plt.show()


def main():
    test(50)


if __name__ == '__main__':
    main()
