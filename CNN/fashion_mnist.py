# -*- coding: utf-8 -*-
# @Time    : 2022/10/18 19:01
# @Author  : 之落花--falling_flowers
# @File    : fashion_mnist.py
# @Software: PyCharm
import torch
import torch.utils.data
import torchvision
import torchvision.datasets
from torch import nn
from torch.nn import functional as f
from torchvision import transforms
import matplotlib.pyplot as plt

import base

PATH = "../pth/fashion_mnist_newnet.pth"


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 4)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.drop1 = nn.Dropout(p=0.2)
        self.drop2 = nn.Dropout(p=0.5)

    def forward(self, x):
        x = f.max_pool2d(f.relu(self.conv1(x)), (2, 2))
        x = f.max_pool2d(f.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = f.relu(self.fc1(self.drop1(x)))
        x = f.relu(self.fc2(self.drop2(x)))
        x = self.fc3(x)
        return x


def train(epoch=1, save=False):
    net = Net()
    net.train(True)
    try:
        net.load_state_dict(torch.load(PATH))
        pass
    except FileNotFoundError:
        pass
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
    trainset = torchvision.datasets.FashionMNIST(root='../data', train=True, download=True, transform=transform)
    trainLoader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
    for i, j in trainLoader:
        base.imgshow(torchvision.utils.make_grid(i))
        print(j)
        input()
        pass
    for e in range(epoch):
        running_loss = 0.0
        i = 0
        x = [x * 4000 for x in range(15)]
        y = []
        for data, target in trainLoader:
            outcome = net(data)
            optimizer.zero_grad()
            loss = criterion(outcome, target)
            loss.backward()
            optimizer.step()
            running_loss += loss
            i += 1
            if i % 1000 == 999:
                y.append(int(running_loss / 1000))
                print(
                    'epoch: %d, complete: (%5d/60000), loss: %.5f' % ((e + 1), (i + 1) * 4, (running_loss / 1000)))
                running_loss = 0.0
        if save:
            torch.save(net.state_dict(), PATH)
        plt.plot(x, y)
        plt.show()


@base.timer
def main():
    train(epoch=3, save=False)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
    testset = torchvision.datasets.FashionMNIST(root='../data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, shuffle=True, num_workers=2)

    base.test(Net(), PATH, testloader)
    pass


if __name__ == '__main__':
    main()
