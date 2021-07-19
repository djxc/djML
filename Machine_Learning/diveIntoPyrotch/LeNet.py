import torch
from torch import nn, optim

from model import LeNet
from data import train_GPU, load_data_fashion_mnist

batch_size = 156
net = LeNet()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_iter, test_iter = load_data_fashion_mnist(batch_size=batch_size)
lr, num_epochs = 0.001, 25
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
train_GPU(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)