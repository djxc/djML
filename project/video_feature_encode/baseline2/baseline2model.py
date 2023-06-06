
from torch import nn

class linear50(nn.Module):
    # 定义模型
    def __init__(self, input_size=2048*50, num_classes=5):
        super().__init__()

        self.name = "linear50"

        self.fc1 = nn.Linear(input_size, 2048)
        self.fc2 = nn.Linear(2048, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x