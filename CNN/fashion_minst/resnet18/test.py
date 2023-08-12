# -*- coding: utf-8 -*-
# @Time    : 2023/7/26 18:18
# @Author  : 之落花--falling_flowers
# @File    : test.py
# @Software: PyCharm
import sys

import torch
from rich.progress import track

from net import Net

sys.path.append('..')
sys.path.append('../../..')

from data import loader
import base

PATH = "D:/Projects/PycharmProjects/Deep-learning/pth\\CNN\\fashion_mnist\\2\\4.pth"
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

testloader = loader(False, resize=True)
net = Net().to(device)
net.eval()

try:
    net.load_state_dict(torch.load(PATH, map_location=device))
except FileNotFoundError as e:
    print(e)
    exit(-1)


@base.ringer
@base.timer
def test():
    i = 0
    for batch in track(testloader, description='Testing...'):
        data, target = [i.to(device) for i in batch]
        outcome = net(data)
        if torch.argmax(outcome) == target[0]:
            i += 1
    print(f'Correct rate: {i}/{len(testloader)}')


def main():
    test()


if __name__ == '__main__':
    main()
