# -*- coding: utf-8 -*-
# @Time    : 2023/8/7 21:52
# @Author  : 之落花--falling_flowers
# @File    : data.py
# @Software: PyCharm
import collections

import torch

import sys
sys.path.append('../../..')
import base


class Vocab:
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []

        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs


def count_corpus(tokens):
    if len(tokens) == 0 or isinstance(tokens[0], list):
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


def read():
    with open(r'D:\Projects\PycharmProjects\Deep-learning\data\Translation\eng-fra.txt', encoding='utf-8') as f:
        return f.readlines()


def load():
    lines = read()
    eng_fra = [i.split('\t') for i in lines]
    eng = [i[0].split(' ') for i in eng_fra]
    fra = [i[1][:-1].split(' ') for i in eng_fra]
    return eng, fra


def build_array(lines, vocab, num_steps):
    array = torch.tensor([truncate_pad(vocab[line] + [vocab['<eos>']], num_steps, vocab['<pad>']) for line in lines])
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
    return array, valid_len


def truncate_pad(line, num_steps, padding_token):
    if len(line) > num_steps:
        return line[:num_steps]
    return line + [padding_token] * (num_steps - len(line))


def batch_reshape(x, batch_size, num_examples):
    if isinstance(x, tuple):
        return tuple(map(lambda a: batch_reshape(a, batch_size, num_examples), x))
    else:
        return x[:-(x.size(0) % batch_size)].reshape(-1, batch_size, *x.size()[1:])[:num_examples]


@base.timer
def load_data(batch_size, num_steps, num_examples=600):
    source, target = load()
    src_vocab = Vocab(source, 2, ['<pad>', '<bos>', '<eos>'])
    tar_vocab = Vocab(target, 2, ['<pad>', '<bos>', '<eos>'])
    src_array, src_valid_len = batch_reshape(build_array(source, src_vocab, num_steps), batch_size, num_examples)
    tar_array, tar_valid_len = batch_reshape(build_array(target, tar_vocab, num_steps), batch_size, num_examples)
    return zip(zip(src_array, src_valid_len), zip(tar_array, tar_valid_len)), src_vocab, tar_vocab
