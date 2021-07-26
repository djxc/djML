import torch

from model import LeNet, AlexNet, VGG, createResNet, createDenseNet
from data import train_GPU, load_data_fashion_mnist
from util import showIMG


def model_demo(net):
    batch_size = 156
    lr, num_epochs = 0.001, 25
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device=torch.device('cpu')
    train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    train_GPU(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)

def create_VGG():
    small_conv_arch = [
        (1, 1, 64),
        (1, 64, 128),
        (2, 128, 256),
        (2, 256, 512),
        (2, 512, 512)
    ]
    fc_features = 512 * 7 * 7
    fc_hidden_units = 4096
    net = VGG(small_conv_arch, fc_features, fc_hidden_units)
    return net

if __name__ == "__main__":
    # net = LeNet()
    # net = AlexNet()
    # net = create_VGG()
    # net = createResNet()
    # net = createDenseNet()
    # print(net)
    # model_demo(net)
    showIMG(bboxs=[[850, 330, 1250, 780], [450, 230, 1050, 480]])
