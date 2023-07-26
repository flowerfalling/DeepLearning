# -*- coding: utf-8 -*-
# @Time    : 2022/10/21 19:49
# @Author  : 之落花--falling_flowers
# @File    : base.py
# @Software: PyCharm
import torch
from torchviz import make_dot


def timer(func):
    import time

    def timerfunc(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f'{func.__name__} used time: {end - start}s')
        return result

    return timerfunc


def ringer(func, beep=(500, 500)):
    import winsound

    def ringfunc(*args, **kwargs):
        result = func(*args, **kwargs)
        winsound.Beep(*beep)
        return result

    return ringfunc


def imgshow(img):
    import numpy as np
    from matplotlib import pyplot as plt
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.ioff()
    plt.show()


def test(net, path, dataloader):
    net.train(False)
    try:
        net.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    except FileNotFoundError as e:
        print(e)
        exit(-1)

    i = 0
    for data, target in dataloader:
        outcome = net(data)
        if torch.argmax(outcome) == target[0]:
            i += 1
    print(f'Correct rate: {i}/{len(dataloader)}')


def summary(input_size, model, _print=True, border=False):
    import pytorchsummary
    pytorchsummary.summary(input_size, model, _print, border)


def imshow(net: torch.nn.Module, input_, format_: str, name: str, directory: str = './image'):
    img = make_dot(net(input_),
                   params=dict(net.named_parameters()),
                   show_attrs=True, show_saved=True)
    img.format = format_
    img.view(cleanup=True, filename=name, directory=directory)
    pass


@timer
def main():
    pass


if __name__ == '__main__':
    main()
