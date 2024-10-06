import torch
from torch import nn


class SyntaxEncoder(nn.Module):
    def __init__(self, output_dim: int):
        super(SyntaxEncoder, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=24, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(in_channels=24, out_channels=64, kernel_size=5),
            nn.MaxPool2d(),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5),
            nn.MaxPool2d(),
            nn.ReLU()
        )
        
        self.ffd = nn.Sequential(
            nn.Linear(128 * 5 * 5, 6400),
            nn.GELU(),
            nn.Linear(6400, 3200),
            nn.GELU(),
            nn.Linear(3200, 1100),
            nn.GELU(),
            nn.Linear(1100, 512),
            nn.GELU(),
            nn.Linear(512, output_dim),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        net = self.cnn(x)
        net = torch.flatten(net, 1)
        
        return self.ffd(net)
