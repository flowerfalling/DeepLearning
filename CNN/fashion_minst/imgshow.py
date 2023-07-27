# -*- coding: utf-8 -*-
# @Time    : 2023/7/26 16:48
# @Author  : 之落花--falling_flowers
# @File    : imgshow.py
# @Software: PyCharm
import torch
from matplotlib import pyplot as plt


def imgshow():
    x = (torch.load('./show.pt') + 1) / 2
    classes = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    label = [0, 6, 8, 6, 5, 4, 6, 2, 1, 8, 6, 4, 0, 8, 0]
    _, axes = plt.subplots(3, 5, figsize=(5, 5))
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, x.reshape(15, 28, 28))):
        ax.imshow(img.numpy())
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_title(classes[label[i]])
    plt.show()


def main():
    imgshow()


if __name__ == '__main__':
    main()
