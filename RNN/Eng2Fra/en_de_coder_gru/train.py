# -*- coding: utf-8 -*-
# @Time    : 2023/8/8 20:40
# @Author  : 之落花--falling_flowers
# @File    : train.py
# @Software: PyCharm
import sys
from typing import Union

import matplotlib.pyplot as plt
import torch
from rich.progress import track
from torch import nn

import net as n

sys.path.append('..')
sys.path.append('../../..')

import data
import base

EPOCH = 100
PATH = "D:\\Projects\\PycharmProjects\\Deep-learning\\pth\\RNN\\Eng2Fra\\en_de_coder_gru\\1.pth"
embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
batch_size, num_steps = 64, 10
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
loss_list = []


def sequence_mask(x, valid_len, value=0):
    maxlen = x.size(1)
    mask = torch.arange(maxlen, dtype=torch.float32, device=x.device)[None, :] < valid_len[:, None]
    x[~mask] = value
    return x


class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super(MaskedCrossEntropyLoss, self).__init__(*args, **kwargs)
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        unweighted_loss = self.criterion(pred.transpose(1, 2), label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss


train_iter, src_vocab, tgt_vocab = data.load_data(batch_size, num_steps)
encoder = n.Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
decoder = n.Seq2SeqDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
net = n.EncoderDecoder(encoder, decoder).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=0.005)
criterion = MaskedCrossEntropyLoss().to(device)
net.train()


@base.ringer
@base.timer
def train(epo=1, load: Union[bool, str] = PATH, save: Union[bool, str] = False):
    global loss_list
    if load:
        try:
            net.load_state_dict(torch.load(load, map_location=device))
        except FileNotFoundError as e:
            print(e)
            exit(-1)
    for epoch in track(range(epo), description='Training...'):
        epo_loss = 0
        for batch in train_iter:
            optimizer.zero_grad()
            x, x_valid_len, t, t_valid_len = [i.to(device) for i in batch]
            bos = torch.tensor([tgt_vocab['<bos>']] * t.shape[0], device=device)[:, None]
            dec_input = torch.cat([bos, t[:, :-1]], 1)
            y, _ = net(x, dec_input, x_valid_len)
            loss = criterion(y, t, t_valid_len).mean()
            epo_loss += loss.to('cpu').detach()
            loss.backward()
            optimizer.step()
        print(f'epoch: {epoch + 1}\tloss: {epo_loss / 600}')
        loss_list.append(epo_loss / 600)
    if save:
        torch.save(net.state_dict(), save)


def main():
    train(EPOCH, False)
    plt.plot(loss_list)
    plt.show()


if __name__ == '__main__':
    main()
