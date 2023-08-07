# -*- coding: utf-8 -*-
# @Time    : 2023/8/2 18:55
# @Author  : 之落花--falling_flowers
# @File    : test.py
# @Software: PyCharm
import sys

import torch

from net import Net

sys.path.append('..')

import data

PATH = "D:\\Projects\\PycharmProjects\\Deep-learning\\pth\\RNN\\The Time Machine\\rnn\\1.pth"
NUM_PREDS = 50


def test(prefix):
    net = Net()
    net.train(False)
    try:
        net.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
    except FileNotFoundError as e:
        print(e)
        exit(-1)
    state = Net.begin_state()
    vocab = data.Vocab(data.tokenize(data.read_time_machine(), 'char'))
    outputs = [vocab[prefix[0]]]
    get_input = lambda: torch.tensor(outputs[-1]).reshape(1, 1)
    for y in prefix[1:]:
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(NUM_PREDS):
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax()))
    print(''.join([vocab.idx_to_token[i] for i in outputs]))


def main():
    test('time traveller')


if __name__ == '__main__':
    main()
