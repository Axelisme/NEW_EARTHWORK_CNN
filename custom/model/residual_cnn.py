
"""A custom neural network model."""

import torch
from torch import nn
from torch import Tensor
from torchvision import models


class ResidualOnly(nn.Module):
    def __init__(self, hidden_size, output_size):
        """Initialize a neural network model."""
        super(ResidualOnly, self).__init__()
        # import pretrain resnet18
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.resnet.fc = nn.Linear(512, hidden_size)
        self.act = nn.GELU()
        self.project = nn.Linear(hidden_size, output_size)

    #@torch.compile
    def forward(self, x:Tensor) -> Tensor:
        """Forward a batch of data through the model."""
        x = self.resnet(x)
        x = self.act(x)
        x = self.project(x)
        return x

