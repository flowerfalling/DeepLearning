# -*- coding: utf-8 -*-
# @Time    : 2023/8/8 19:44
# @Author  : 之落花--falling_flowers
# @File    : net.py
# @Software: PyCharm
import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Encoder, self).__init__(*args, **kwargs)

    def forward(self, x):
        raise NotImplementedError


class Decoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Decoder, self).__init__(*args, **kwargs)

    def forward(self, x, state):
        raise NotImplementedError

    def init_state(self, en_outputs):
        raise NotImplementedError


class Seq2SeqEncoder(Encoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, num_hiddens, num_layers, dropout=dropout, batch_first=True)

    def forward(self, x, *args):
        x = self.embedding(x)
        y, state = self.gru(x)
        return y, state


class Seq2SeqDecoder(Decoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(num_hiddens, vocab_size)

    def forward(self, x, state, *args):
        x = self.embedding(x)
        x = torch.cat((x, state[-1].repeat(x.shape[1], 1, 1).transpose(0, 1)), 2)
        y, state = self.gru(x, state)
        y = self.fc(y)
        return y, state

    def init_state(self, en_outputs, *args):
        return en_outputs[1]


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)

