# -*- coding: utf-8 -*-
# @Time    : 2022/10/4 18:21
# @Author  : 之落花--falling_flowers
# @File    : mnist.py
# @Software: PyCharm
import torch
import torch.utils.data
import torchvision.datasets
from torch import nn
from torch.nn import functional as f
from torchvision import transforms

import base

PATH = r"D:\Projects\PycharmProjects\Digital-recognition-GUI\net.pth"


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 4)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = f.max_pool2d(f.relu(self.conv1(x)), (2, 2))
        x = f.max_pool2d(f.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(times, save=False):
    net = Net()
    try:
        net.load_state_dict(torch.load(PATH))
        # pass
    except FileNotFoundError:
        pass
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
    trainset = torchvision.datasets.MNIST(root='../data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

    for epoch in range(times):
        running_loss = 0.0
        i = 0

        for data, target in trainloader:
            outcome = net(data)
            optimizer.zero_grad()
            loss = criterion(outcome, target)
            loss.backward()
            optimizer.step()
            running_loss += loss
            i += 1
            if i % 1000 == 999:
                print(
                    'epoch: %d, complete: (%5d/60000), loss: %.5f' % ((epoch + 1), (i + 1) * 4, (running_loss / 1000)))
                running_loss = 0.0

    if save:
        torch.save(net.state_dict(), PATH)


def test():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
    testset = torchvision.datasets.MNIST(root='../data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, shuffle=True, num_workers=2)
    base.test(Net(), PATH, testloader)


@base.ringer
@base.timer
def main():
    # train(2, True)
    # base.imshow(Net(), torch.randn(64, 1, 28, 28), 'png', 'mnist', './image')
    test()
    # net = Net()
    # net.load_state_dict(torch.load(PATH))
    # torch.save(net, 'net.pkl')
    pass


if __name__ == '__main__':
    main()
