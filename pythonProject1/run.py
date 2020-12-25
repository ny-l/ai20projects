import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import os
path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)

class MNIST_Net(nn.Module):
    def __init__(self, N=10):
        super(MNIST_Net, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1,  6, 5),
            nn.MaxPool2d(2, 2), # 6 24 24 -> 6 12 12
            nn.ReLU(True),
            nn.Conv2d(6, 16, 5), # 6 12 12 -> 16 8 8
            nn.MaxPool2d(2, 2), # 16 8 8 -> 16 4 4
            nn.ReLU(True)
        )
        self.classifier =  nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, N),
            nn.Softmax(1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, 16 * 4 * 4)
        x = self.classifier(x)
        return x


def test_MNIST(model,max_digit=10,name='mnist_net'):
    confusion = np.zeros((max_digit,max_digit),dtype=np.uint32) # First index actual, second index predicted
    N = 0
    for d,l in mnist_test_data:
        if l < max_digit:
            N += 1
            d = Variable(d.unsqueeze(0))
            outputs = model.networks[name].net.forward(d)
            _, out = torch.max(outputs.data, 1)
            c = int(out.squeeze())
            confusion[l,c] += 1
    print(confusion)
    F1 = 0
    for nr in range(max_digit):
        TP = confusion[nr,nr]
        FP = sum(confusion[:,nr])-TP
        FN = sum(confusion[nr,:])-TP
        F1 += 2*TP/(2*TP+FP+FN)*(FN+TP)/N
    print('F1: ',F1)
    return [('F1',F1)]


def neural_predicate(network, i):
    dataset = str(i.functor)
    i = int(i.args[0])
    if dataset == 'train':
        d, l = mnist_train_data[i]
    elif dataset == 'test':
        d, l = mnist_test_data[i]
    d = Variable(d.unsqueeze(0))
    output = network.net(d)
    return output.squeeze(0)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5, ))])
mnist_train_data = torchvision.datasets.MNIST(root=dir_path+'/../../../data/MNIST', train=True, download=True,transform=transform)
mnist_test_data = torchvision.datasets.MNIST(root=dir_path+'/../../../data/MNIST', train=False, download=True,transform=transform)


import torch
import torchvision
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from logger import Logger
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.MaxPool2d(2, 2),  # 6 24 24 -> 6 12 12
            nn.ReLU(True),
            nn.Conv2d(6, 16, 5),  # 6 12 12 -> 16 8 8
            nn.MaxPool2d(2, 2),  # 16 8 8 -> 16 4 4
            nn.ReLU(True)
        )
        self.classifier = nn.Sequential(

            nn.Linear(16 * 11 * 4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 19),
            nn.LogSoftmax(1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, 16 * 11 * 4)
        x = self.classifier(x)
        return x


def test_MNIST():
    confusion = np.zeros((19, 19), dtype=np.uint32)  # First index actual, second index predicted
    correct = 0
    n = 0
    N = len(test_dataset)
    for d, l in test_dataset:
        d = Variable(d.unsqueeze(0))
        outputs = net.forward(d)
        _, out = torch.max(outputs.data, 1)
        c = int(out.squeeze())
        confusion[l, c] += 1
        if c == l:
            correct += 1
        n += 1
    acc = correct / n
    print(confusion)
    F1 = 0
    for nr in range(19):
        TP = confusion[nr, nr]
        FP = sum(confusion[:, nr]) - TP
        FN = sum(confusion[nr, :]) - TP
        F1 += 2 * TP / (2 * TP + FP + FN) * (FN + TP) / N
    print('F1: ', F1)
    print('Accuracy: ', acc)
    return F1


net = Net()
optimizer = optim.Adam(net.parameters(), lr=0.001)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
criterion = nn.NLLLoss()


class MNIST_Addition(Dataset):

    def __init__(self, dataset, examples):
        self.data = list()
        self.dataset = dataset
        with open(examples) as f:
            for line in f:
                line = line.strip().split(' ')
                self.data.append(tuple([int(i) for i in line]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        i1, i2, l = self.data[index]
        return torch.cat((self.dataset[i1][0], self.dataset[i2][0]), 1), l


train_dataset = MNIST_Addition(
    torchvision.datasets.MNIST(root='../../../data/MNIST', train=True, download=True, transform=transform),
    'train_data.txt')
test_dataset = MNIST_Addition(
    torchvision.datasets.MNIST(root='../../../data/MNIST', train=False, download=True, transform=transform),
    'test_data.txt')
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=1)

i = 1
test_period = 500
log_period = 50
running_loss = 0.0
log = Logger()

for epoch in range(1):

    for data in trainloader:
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.data[0]
        if i % log_period == 0:
            print('Iteration: ', i * 2, '\tAverage Loss: ', running_loss / log_period)
            log.log('loss', i * 2, running_loss / log_period)
            running_loss = 0
        if i % test_period == 0:
            log.log('F1', i * 2, test_MNIST())
        i += 1



from train import train_model
from data_loader import load
from examples.NIPS.MNIST.mnist import test_MNIST, MNIST_Net, neural_predicate
from model import Model
from optimizer import Optimizer
from network import Network
import torch

queries = load('train_data.txt')
test_queries = load('test_data.txt')

with open('addition.pl') as f:
    problog_string = f.read()

network = MNIST_Net()
net = Network(network, 'mnist_net', neural_predicate)
net.optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
model = Model(problog_string, [net], caching=False)
optimizer = Optimizer(model, 2)

train_model(model, queries, 1, optimizer, test_iter=1000, test=lambda x: x.accuracy(test_queries, test=True), snapshot_iter=10000)