# -*- coding: utf-8 -*-
# @Time    : 2023/8/2 11:38
# @Author  : 之落花--falling_flowers
# @File    : net.py
# @Software: PyCharm
import torch
from torch import nn
from torch.nn import functional as F


class Net(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Net, self).__init__(*args, **kwargs)
        self.vocab_size = 28
        self.gru = nn.GRU(
            input_size=self.vocab_size,
            hidden_size=256,
            batch_first=True
        )
        self.fc = nn.Linear(256, self.vocab_size)

    def forward(self, inputs, state):
        x = F.one_hot(inputs.long(), self.vocab_size).to(torch.float32)
        y, state = self.gru(x, state)
        output = self.fc(y)
        return output, state

    @staticmethod
    def begin_state(batch_size=1):
        return torch.zeros((1, batch_size, 256))
