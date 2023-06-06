from torchvision import transforms
import torchvision.datasets as dst

from encoder_config import workspace_root

def get_mnist_data():
    transform = transforms.Compose([transforms.ToTensor()])
    data_train = dst.MNIST(workspace_root, train=True, transform=transform, download=True)
    data_test = dst.MNIST(workspace_root, train=False, transform=transform)
    return data_train, data_test
