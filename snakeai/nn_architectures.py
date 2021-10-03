from torch import nn
from torch.nn import functional


class NLinearNet(nn.Module):

    def __init__(self, n, units, use_sigmoid=False):
        super().__init__()
        if n + 1 != len(units):
            raise ValueError(f"expected {n} layers, got {len(units) - 1}")
        self.layers = [nn.Linear(units[i - 1], units[i]) for i in range(1, n)]
        self.architecture = str(units)
        self.use_sigmoid = use_sigmoid

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = functional.relu(layer(x))
        x = self.layers[-1](x)
        if self.use_sigmoid:
            x = functional.sigmoid(x)
        return x
