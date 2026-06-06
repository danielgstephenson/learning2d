import torch
from torch import nn, Tensor
from math import log
import torch.nn.functional as F

state_size = 34

a0vel  = list(range(0,  2))
a1vel  = list(range(2,  4))
b0vel  = list(range(4,  6))
b1vel  = list(range(6,  8))
a0pos  = list(range(8,  10))
a1pos  = list(range(10, 12))
b0pos  = list(range(12, 14))
b1pos  = list(range(14, 16))
alive0 = [16]
alive1 = [17]
wp0    = list(range(18, 26))
wp1    = list(range(26, 34))
swap_idx = \
    a1vel + a0vel + b1vel + b0vel + a1pos + a0pos + \
    b1pos + b0pos + alive1 + alive0 + wp1 + wp0

class SwapState(nn.Module):
    idx: Tensor
    def __init__(self):
        super().__init__()
        self.register_buffer('idx', torch.tensor(swap_idx))
    def forward(self, x: Tensor) -> Tensor:
        return x[..., self.idx]

class ValueModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_dim = state_size
        k = 512
        layer_count = 4
        self.projection = nn.Linear(self.input_dim, k)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(k) for _ in range(layer_count)])
        self.hidden_layers = nn.ModuleList([nn.Linear(k, k) for _ in range(layer_count)])
        self.output_layer = nn.Linear(k, 1)
        self.final_norm = nn.LayerNorm(k)
    def forward(self, x: Tensor) -> Tensor:
        x = self.projection(x)
        for norm, layer in zip(self.layer_norms, self.hidden_layers):
            x = x + layer(F.celu(norm(x)))
        return self.output_layer(self.final_norm(x))
    def __call__(self, *args, **kwds)->Tensor:
        return super().__call__(*args, **kwds)

    
