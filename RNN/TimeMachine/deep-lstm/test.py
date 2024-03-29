# -*- coding: utf-8 -*-
# @Time    : 2023/8/2 18:55
# @Author  : 之落花--falling_flowers
# @File    : test.py
# @Software: PyCharm
import torch

from RNN.TimeMachine import data
from net import Net

NUM_PREDS = 50
PATH = r"D:\Projects\PycharmProjects\DeepLearning\pth\RNN\TimeMachine\deep-lstm\1.pth"
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

net = Net().to(device)
net.eval()

try:
    net.load_state_dict(torch.load(PATH, map_location=device))
except FileNotFoundError as e:
    print(e)
    exit(-1)


def test(prefix):
    state = [i.to(device) for i in Net.begin_state()]
    vocab = data.Vocab(data.tokenize(data.read_time_machine(), 'char'))
    outputs = [vocab[prefix[0]]]
    for y in prefix[1:]:
        _, state = net(torch.tensor(outputs[-1], device=device).reshape(1, 1), state)
        outputs.append(vocab[y])
    for _ in range(NUM_PREDS):
        y, state = net(torch.tensor(outputs[-1], device=device).reshape(1, 1), state)
        outputs.append(int(y.argmax()))
    return ''.join([vocab.idx_to_token[i] for i in outputs])


def main():
    print(test('time traveller'))


if __name__ == '__main__':
    main()
