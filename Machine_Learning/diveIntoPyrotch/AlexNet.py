import torch

from model import AlexNet
from data import train_GPU, load_data_fashion_mnist

net = AlexNet()
print(net)

batch_size = 128
# 如出现“out of memory”的报错信息,可减小小batch_size或resize
train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224)

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr, num_epochs = 0.001, 15
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
train_GPU(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)