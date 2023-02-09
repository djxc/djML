
import os
import torch.nn as nn
import torch, torchvision

from config import device, model_root


class LandUseNet():
    def __init__(self, net_name: str, class_num: int, resume=False) -> None:
        self.net_name = net_name
        self.class_num = class_num
        self.resume = resume
        self.model_folder = r"{}\{}".format(model_root, net_name)
        self.net = self.__create_net()
        if self.resume:
            self.load()
        if not os.path.exists(self.model_folder):
            os.mkdir(self.model_folder)


    def save(self, model_name):
        """保存模型"""
        model_path = os.path.join(self.model_folder, model_name)
        torch.save(self.net.state_dict(), model_path)

    def load(self):
        """加载模型的参数"""
        best_model_path = "{}\\best_model.pth".format(self.model_folder)
        if os.path.exists(best_model_path):
            print("加载模型。。。")
            self.net.load_state_dict(torch.load(best_model_path))


    def __create_net(self):
        """根据模型名称创建模型
        """
        net_name = self.net_name
        class_num = self.class_num
        print("create {} net ....".format(net_name))
        if net_name == "efficientNet":
            net = EfficientNet.from_pretrained('efficientnet-b4',  num_classes=class_num)
            # net = EfficientNet.from_name('efficientnet-b4',  num_classes=176)
        elif net_name == "resNet":
            net = createResNet()
        elif net_name == "resNet50_pre":
            net = torchvision.models.resnet50(pretrained=True)
            # set_parameter_requires_grad(model_ft, False)  # 固定住前面的网络层
            num_ftrs = net.fc.in_features
            # 修改最后的全连接层
            net.fc = nn.Sequential(
                nn.Linear(num_ftrs, class_num)
            )
        elif net_name == "resnext":
            net = torchvision.models.resnext50_32x4d(pretrained=True)
            # set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = net.fc.in_features
            net.fc = nn.Sequential(nn.Linear(num_ftrs, class_num))
        net = net.to(device)      
        return net