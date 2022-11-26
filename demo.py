# -*- coding: utf-8 -*-
# @Time    : 2022/10/4 21:52
# @Author  : 之落花--falling_flowers
# @File    : demo.py
# @Software: PyCharm
import torch
import torch.utils.data
import torchvision
from torch import nn
from torchvision import transforms

import base


class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(Net, self).__init__()
        self.hidden_dim = hidden_dim
        self.later_dim = layer_dim
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h = torch.zeros(self.later_dim, x.size[0], self.hidden_dim).requires_grad()


def train():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)


def test():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=True, num_workers=2)


@base.timer
def main(s: str, word_dict: list) -> bool:
    lenght = len(s)
    state = [False for _ in range(lenght + 1)]
    state[0] = True
    for i in range(1, lenght + 1):
        for j in range(i):
            if s[j:i] in word_dict:
                state[i] = state[j]
            if state[i]:
                break
    return state[lenght]


if __name__ == '__main__':
    print(main('catsandog', ['cats', 'send', 'dog', 'and', 'cat']))
    pass
