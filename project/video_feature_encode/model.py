
from torch import nn
from torch.nn import functional as F

## 使用多层感知机，数据量太大不能全部放入内存 #
class MLPModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512000, 25600),
            nn.ReLU(),
            # nn.Linear(256000, 102400),
            # nn.ReLU(),
            # nn.Linear(102400, 25600),
            # nn.ReLU(),
            # nn.Linear(51200, 25600),
            # nn.ReLU(),
            nn.Linear(25600, 5120),
            nn.ReLU(),
            # nn.Linear(10240, 5120),
            # nn.ReLU(),
            nn.Linear(5120, 1024),
            nn.ReLU(),
            # nn.Linear(2560, 1024),
            # nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
        )
        self.out = nn.Linear(256, 5)  # 输出层

    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))
