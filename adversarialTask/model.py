import torch
from torch import nn

class WhiteBoxModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 24, 5, 1, 2),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(24, 48, 5, 2, 2),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Conv2d(48, 64, 5, 3, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(5*5*64, 1024),
            nn.ReLU(),
            nn.Linear(1024, 10)
        )
    
    def forward(self, x):
        x = x.reshape((x.size(0), 1, 28, 28))
        return self.layers(x)
