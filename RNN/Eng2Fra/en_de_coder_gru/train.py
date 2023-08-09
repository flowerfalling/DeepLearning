# -*- coding: utf-8 -*-
# @Time    : 2023/8/8 20:40
# @Author  : 之落花--falling_flowers
# @File    : train.py
# @Software: PyCharm
import sys

import torch
from rich.progress import track
from torch import nn

import net as n

sys.path.append('..')
sys.path.append('../../..')

import data
import base

PATH = "D:\\Projects\\PycharmProjects\\Deep-learning\\pth\\RNN\\Eng2Fra\\en_de_coder_gru\\1.pth"
EPOCH = 100

embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
batch_size, num_steps = 64, 10


def sequence_mask(X, valid_len, value=0):
    maxlen = X.size(1)
    mask = torch.arange(maxlen, dtype=torch.float32, device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X


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


def train(epo=1, load=True, save=False, save_path=PATH):
    train_iter, src_vocab, tgt_vocab = data.load_data(batch_size, num_steps)
    encoder = n.Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
    decoder = n.Seq2SeqDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
    net = n.EncoderDecoder(encoder, decoder)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.005)
    criterion = MaskedCrossEntropyLoss()
    train_iter = list(train_iter)
    for epoch in track(range(epo), description='Training...'):
        epo_loss = 0
        for (x, x_valid_len), (t, t_valid_len) in train_iter:
            optimizer.zero_grad()
            bos = torch.tensor([tgt_vocab['<bos>']] * t.shape[0])[:, None]
            dec_input = torch.cat([bos, t[:, :-1]], 1)
            y, _ = net(x, dec_input, x_valid_len)
            loss = criterion(y, t, t_valid_len).mean()
            epo_loss += loss
            loss.backward()
            optimizer.step()
        print(f'epoch: {epoch + 1}\tloss: {epo_loss / 600}')
    if save:
        torch.save(net.state_dict(), save_path)


@base.timer
def main():
    train(EPOCH, False, True)


if __name__ == '__main__':
    main()
