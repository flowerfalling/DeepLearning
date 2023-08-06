# -*- coding: utf-8 -*-
# @Time    : 2023/8/6 17:54
# @Author  : 之落花--falling_flowers
# @File    : net.py
# @Software: PyCharm
from torch import nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.rnn = nn.RNN(
            input_size=4,
            hidden_size=32,
            batch_first=True
        )
        self.fc = nn.Linear(32, 1)

    def forward(self, x, h):
        out, hidden_prev = self.rnn(x, h)
        out = out.reshape(-1, 32)
        out = self.fc(out)
        out = out.unsqueeze(dim=0)
        return out, hidden_prev
