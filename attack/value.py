from torch import nn, Tensor
import torch.nn.functional as F
from physics import physics_dtype

class ValueModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_dim = 14
        k = 512
        self.projection = nn.Linear(self.input_dim, k)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(k) for _ in range(4)])
        self.hidden_layers = nn.ModuleList([nn.Linear(k, k) for _ in range(4)])
        self.output_layer = nn.Linear(k, 1)
    def forward(self, x: Tensor)->Tensor:
        x = self.projection(x)
        for norm, layer in zip(self.layer_norms, self.hidden_layers):
            x = x + layer(F.silu(norm(x)))
        return self.output_layer(x)
    def __call__(self, *args, **kwds)->Tensor:
        return super().__call__(*args, **kwds)
    
