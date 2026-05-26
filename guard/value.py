import torch
from torch import nn, Tensor
import torch.nn.functional as F

state_size = 33
    
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
    def forward(self, x: Tensor)->Tensor:
        x = 0.01 * self.projection(x)
        for norm, layer in zip(self.layer_norms, self.hidden_layers):
            x = x + layer(F.silu(norm(x)))
        return self.output_layer(self.final_norm(x))
    def __call__(self, *args, **kwds)->Tensor:
        return super().__call__(*args, **kwds)
    
