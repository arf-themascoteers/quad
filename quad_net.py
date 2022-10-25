import torch.nn as nn
import torch.nn.functional as F


class QuadNet(nn.Module):
    def __init__(self):
        super(QuadNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        x = self.net(x)
        return x