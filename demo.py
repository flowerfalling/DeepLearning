# -*- coding: utf-8 -*-
# @Time    : 2022/10/4 21:52
# @Author  : 之落花--falling_flowers
# @File    : demo.py
# @Software: PyCharm
# from bisect import bisect_left
#
#
# def main():
#     try:
#         weight = float(input('体重(kg):'))
#         height = float(input('身高(m):'))
#     except ValueError as e:
#         print('瞎jb输入nm呢\n', e)
#         return 5
#     sex = dict((('男', 1), ('女', 0))).get(input('性别(男(默认)/女):'), 1)
#     bmi = round(weight / height ** 2, 1)
#     print('bmi: ', bmi)
#     a = ((0, 16.5, 22.8, 25.4), (0, 16.6, 23.3, 26.5))
#     return bisect_left(a[sex], bmi)
#
#
# if __name__ == '__main__':
#     print(['您', '低体重', '正常', '超重', '肥胖', '爬'][main()])
from torch import nn
from torch.nn import functional as f
import torch


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
# net.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
script = torch.jit.trace(net, torch.randn(1, 28, 28))
script.save('net.pt')
