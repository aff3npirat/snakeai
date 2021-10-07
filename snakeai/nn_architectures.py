import torch.nn as nn
import torch.nn.functional as functional


class LinearNet(nn.Module):

    def __init__(self, architecture, use_sigmoid=False):
        super().__init__()
        self.layers = [nn.Linear(architecture[i - 1], architecture[i]) for i in range(1, len(architecture))]
        self.use_sigmoid = use_sigmoid

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = functional.relu(layer(x))
        x = self.layers[-1](x)
        if self.use_sigmoid:
            x = functional.sigmoid(x)
        return x
