# -*- coding: utf-8 -*-
# @Time    : 2023/7/26 16:48
# @Author  : 之落花--falling_flowers
# @File    : imgshow.py
# @Software: PyCharm
import torch
from matplotlib import pyplot as plt


def imgshow():
    x = (torch.load('./show.pt') + 1) / 2
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    label = [6, 1, 0, 9, 7, 1, 8, 9, 3, 5, 6, 7, 1, 8, 9]
    _, axes = plt.subplots(3, 5, figsize=(5, 5))
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, x.reshape(15, 3, 32, 32))):
        ax.imshow(img.numpy().transpose([1, 2, 0]))
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_title(classes[label[i]])
    plt.show()
    pass


def main():
    imgshow()


if __name__ == '__main__':
    main()
