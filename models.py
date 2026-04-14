from math import sqrt

from torch import nn, Tensor
import torch.nn.functional as F

class ValueModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_dim = 18
        k = 200
        self.hidden_count = 4
        self.scale_factor = 1 / sqrt(self.hidden_count)
        self.projection_layer = nn.Linear(self.input_dim, k)
        self.hidden_layers = nn.ModuleList()
        for _ in range(self.hidden_count):
            self.hidden_layers.append(nn.Linear(k, k))
        self.output_layer = nn.Linear(k, 1)
    def forward(self, x: Tensor)->Tensor:
        x = self.projection_layer(x)
        for i in range(self.hidden_count):
            h = self.hidden_layers[i]
            x = x + F.leaky_relu(h(x),negative_slope=0.01) * self.scale_factor
        x = self.output_layer(x)
        return x
    def __call__(self, *args, **kwds)->Tensor:
        return super().__call__(*args, **kwds)

class ActionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_dim = 18
        k = 50
        self.hidden_count = 4
        self.projection_layer = nn.Linear(self.input_dim, k)
        self.hidden_layers = nn.ModuleList()
        for _ in range(self.hidden_count):
            self.hidden_layers.append(nn.Linear(k, k))
        self.output_layer = nn.Linear(k, 9)
    def forward(self, x: Tensor)->Tensor:
        x = self.projection_layer(x)
        for i in range(self.hidden_count):
            h = self.hidden_layers[i]
            x = x + F.leaky_relu(h(x),negative_slope=0.01)
        x = self.output_layer(x)
        return x
    def __call__(self, *args, **kwds)->Tensor:
        return super().__call__(*args, **kwds)
    
