from torch import nn
from torch.nn import functional


class NLinearNet(nn.Module):

    def __init__(self, units, use_sigmoid=False):
        super().__init__()
        self.layers = [nn.Linear(units[i - 1], units[i]) for i in range(1, len(units))]
        self.architecture = str(units)
        self.use_sigmoid = use_sigmoid

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = functional.relu(layer(x))
        x = self.layers[-1](x)
        if self.use_sigmoid:
            x = functional.sigmoid(x)
        return x
