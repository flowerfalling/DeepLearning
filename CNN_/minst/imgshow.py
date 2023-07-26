# -*- coding: utf-8 -*-
# @Time    : 2023/7/25 22:39
# @Author  : 之落花--falling_flowers
# @File    : imgshow.py
# @Software: PyCharm
import torch
from matplotlib import pyplot as plt


def imgshow():
    x = torch.load('./show.pt')
    _, axes = plt.subplots(3, 4, figsize=(3, 5))
    axes = axes.flatten()
    for ax, img in zip(axes, x.reshape(15, 28, 28)):
        ax.imshow(img.numpy())
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
    plt.show()


def main():
    imgshow()


if __name__ == '__main__':
    main()
