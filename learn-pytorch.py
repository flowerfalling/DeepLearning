# -*- coding: utf-8 -*-
# @Time    : 2022/12/23 20:33
# @Author  : 之落花--falling_flowers
# @File    : learn-pytorch.py
# @Software: PyCharm
import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter


def main():
    writer = SummaryWriter('logs')
    # for i in range(100):
    #     writer.add_scalar('y=x', i ** 2, i)
    image_path = r"D:\falling\Pictures\DIno的插图・漫画 - pixiv\96492892_p0.jpg"
    writer.add_image('img', np.array(Image.open(image_path)), 1, dataformats='HWC')
    writer.close()
    pass


if __name__ == '__main__':
    main()
