# -*- coding: utf-8 -*-
# @Time    : 2022/10/21 19:49
# @Author  : 之落花--falling_flowers
# @File    : base.py
# @Software: PyCharm
import torch


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
    plt.show(ioff=True)


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


def make_dot(var, params=None):
    """
    Produces Graphviz representation of PyTorch autograd graph
    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function
    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad
    """
    from graphviz import Digraph
    from torch.autograd import Variable
    if params is not None:
        assert isinstance(params.values()[0], Variable)
        param_map = {id(v): k for k, v in params.items()}

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(format='png', node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def size_to_str(size):
        return '(' + ', '.join(['%d' % v for v in size]) + ')'

    def add_nodes(var_):
        if var_ not in seen:
            if torch.is_tensor(var_):
                dot.node(str(id(var_)), size_to_str(var_.size()), fillcolor='orange')
            elif hasattr(var_, 'variable'):
                u = var_.variable
                name = param_map[id(u)] if params is not None else ''
                node_name = '%s\n %s' % (name, size_to_str(u.size()))
                dot.node(str(id(var_)), node_name, fillcolor='lightblue')
            else:
                dot.node(str(id(var_)), str(type(var_).__name__))
            seen.add(var_)
            if hasattr(var_, 'next_functions'):
                for u in var_.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var_)))
                        add_nodes(u[0])
            if hasattr(var_, 'saved_tensors'):
                for t in var_.saved_tensors:
                    dot.edge(str(id(t)), str(id(var_)))
                    add_nodes(t)

    add_nodes(var.grad_fn)
    return dot


def net_image(net, input_size, name='net'):
    x = Variable(torch.randn(*input_size))
    y = net(x)
    g = make_dot(y)
    g.render(filename=name, view=False, cleanup=True)

    params = list(net.parameters())
    k = 0
    for i in params:
        c = 1
        print("该层的结构：" + str(list(i.size())))
        for j in i.size():
            c *= j
        print("该层参数和：" + str(c))
        k = k + c
    print("总参数数量和：" + str(k))


def summary(input_size, model, _print=True, border=False):
    import pytorchsummary
    pytorchsummary.summary(input_size, model, _print, border)


@timer
def main():
    pass


if __name__ == '__main__':
    main()
