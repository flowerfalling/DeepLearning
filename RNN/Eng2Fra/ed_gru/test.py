# -*- coding: utf-8 -*-
# @Time    : 2023/8/9 14:56
# @Author  : 之落花--falling_flowers
# @File    : test.py
# @Software: PyCharm
import torch

import net as n
from RNN.Eng2Fra import data

PATH = r"D:\Projects\PycharmProjects\DeepLearning\pth\RNN\Eng2Fra\en_de_coder_gru\1.pth"

embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
batch_size, num_steps = 64, 10
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
train_iter, src_vocab, tgt_vocab = data.load_data(batch_size, num_steps)
encoder = n.Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
decoder = n.Seq2SeqDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
net = n.EncoderDecoder(encoder, decoder).to(device)
net.eval()

try:
    net.load_state_dict(torch.load(PATH, map_location=device))
except FileNotFoundError as e:
    print(e)
    exit(-1)


def test(src_sentence):
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [src_vocab['<eos>']]
    enc_valid_len = torch.tensor([len(src_tokens)])
    src_tokens = data.truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    # 添加批量轴
    enc_X = torch.unsqueeze(torch.tensor(src_tokens, dtype=torch.long), dim=0)
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    # 添加批量轴
    dec_X = torch.unsqueeze(torch.tensor([tgt_vocab['<bos>']], dtype=torch.long), dim=0)
    output_seq = []
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        # 我们使用具有预测最高可能性的词元，作为解码器在下一时间步的输入
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        # 一旦序列结束词元被预测，输出序列的生成就完成了
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq))


def main():
    for i in ['go .', "i lost .", 'he \'s calm .', 'i \'m home .']:
        print(test(i))


if __name__ == '__main__':
    main()
