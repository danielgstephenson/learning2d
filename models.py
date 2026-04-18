import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F

class ValueModel(nn.Module):
    projection_weight: Tensor
    projection_bias: Tensor
    def __init__(self):
        super().__init__()
        self.input_dim = 26
        k = 512
        k_band = 128
        stds = [1.0, 5.0, 25.0, 100.0]
        weight_list = []
        for s in stds:
            w = torch.randn(k_band, self.input_dim) * s
            weight_list.append(w)
        self.register_buffer('projection_weight', torch.cat(weight_list, dim=0))
        self.register_buffer('projection_bias', torch.rand(k) * 2 * math.pi)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(k) for _ in range(4)])
        self.hidden_layers = nn.ModuleList([nn.Linear(k, k) for _ in range(4)])
        self.output_layer = nn.Linear(k, 1)
        nn.init.constant_(self.output_layer.bias, -15.0)
    def forward(self, x: Tensor)->Tensor:
        x = torch.sin(F.linear(x, self.projection_weight, self.projection_bias))
        for norm, layer in zip(self.layer_norms, self.hidden_layers):
            x = x + layer(F.silu(norm(x)))
        return self.output_layer(x)
    def __call__(self, *args, **kwds)->Tensor:
        return super().__call__(*args, **kwds)
    
class GradientModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_dim = 26
        k = 128
        self.projection = nn.Linear(self.input_dim, k)
        nn.init.normal_(self.projection.weight, mean=0.0, std=2.0)
        nn.init.uniform_(self.projection.bias, 0, 2 * math.pi)
        self.projection.weight.requires_grad = False
        self.projection.bias.requires_grad = False
        self.layer_norms = nn.ModuleList([nn.LayerNorm(k) for _ in range(4)])
        self.hidden_layers = nn.ModuleList([nn.Linear(k, k) for _ in range(4)])
        self.output_layer = nn.Linear(k, 2)
        nn.init.constant_(self.output_layer.bias, -0.0)
    def forward(self, x: Tensor)->Tensor:
        x = torch.sin(self.projection(x))
        for norm, layer in zip(self.layer_norms, self.hidden_layers):
            x = x + layer(F.silu(norm(x)))
        return self.output_layer(x)
    def __call__(self, *args, **kwds)->Tensor:
        return super().__call__(*args, **kwds)
    
