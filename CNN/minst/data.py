# -*- coding: utf-8 -*-
# @Time    : 2023/7/25 21:13
# @Author  : 之落花--falling_flowers
# @File    : data.py
# @Software: PyCharm
import torch.utils.data
import torchvision
from torchvision import transforms

DATA_PATH = r"D:\Projects\PycharmProjects\DeepLearning\data"


def loader(train=True, batch_size=1):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
    data_set = torchvision.datasets.MNIST(root=DATA_PATH, train=train, download=True, transform=transform)
    set_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=True, num_workers=2)
    return set_loader
