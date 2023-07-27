# -*- coding: utf-8 -*-
# @Time    : 2023/7/26 16:17
# @Author  : 之落花--falling_flowers
# @File    : test.py
# @Software: PyCharm
import sys

import torch

from net import Net

sys.path.append('..')
sys.path.append('../../..')

from data import loader
import base

PATH = "D:/Projects/PycharmProjects/Deep-learning/pth\\CNN\\cifar-10\\1\\1.pth"


def test():
    net = Net()
    net.train(False)
    try:
        net.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
    except FileNotFoundError as e:
        print(e)
        exit(-1)
    testloader = loader(False)
    i = 0
    for data, target in testloader:
        outcome = net(data)
        if torch.argmax(outcome) == target[0]:
            i += 1
    print(f'Correct rate: {i}/{len(testloader)}')


@base.ringer
@base.timer
def main():
    test()


if __name__ == '__main__':
    main()
