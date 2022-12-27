# -*- coding: utf-8 -*-
# @Time    : 2022/10/15 10:23
# @Author  : 之落花--falling_flowers
# @File    : cifar-10.py.py
# @Software: PyCharm
import torch
import torch.utils.data
import torchvision
import torchvision.datasets
from torch import nn
from torch.nn import functional as f
from torchvision import transforms

# import base

PATH = "../pth/cifar_net.pth"


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 3)
        self.conv2 = nn.Conv2d(20, 100, 4)
        self.conv3 = nn.Conv2d(100, 500, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(2000, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 500)
        self.fc4 = nn.Linear(500, 200)
        self.fc5 = nn.Linear(200, 100)
        self.fc6 = nn.Linear(100, 10)

    def forward(self, x):
        y = self.pool(f.relu(self.conv1(x)))
        y = self.pool(f.relu(self.conv2(y)))
        y = self.pool(f.relu(self.conv3(y)))
        y = y.reshape(-1, 2000)
        y = f.relu(self.fc1(y))
        y = f.relu(self.fc2(y))
        y = f.relu(self.fc3(y))
        y = f.relu(self.fc4(y))
        y = f.relu(self.fc5(y))
        y = self.fc6(y)
        return y


def train(times, save=False):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, 4)])
    trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

    net = Net()
    try:
        # net.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
        pass
    except FileNotFoundError:
        pass
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(times):
        running_loss = 0.0
        i = 0
        for j in trainloader:
            data, target = j[0], j[1]
            outcome = net(data)
            optimizer.zero_grad()
            loss = criterion(outcome, target)
            loss.backward()
            optimizer.step()
            running_loss += loss
            i += 1
            if i % 500 == 499:
                print(
                    'epoch: %d, complete: (%5d/50000), loss: %.5f' % ((epoch + 1), (i + 1) * 4, (running_loss / 500)))
                running_loss = 0.0

        if save:
            torch.save(net.state_dict(), PATH)


# @base.ringer
# @base.timer
def main():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, shuffle=False, num_workers=2)
    train(1)
    # base.test(Net(), PATH, testloader)
    pass


if __name__ == '__main__':
    main()
    # train(3, False)
    # test()
    # winsound.Beep(500, 1000)
