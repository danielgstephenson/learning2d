import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from physics import physics_dtype

bucket_count = 101
bucket_indices = torch.tensor([i for i in range(bucket_count)])
boundaries = torch.linspace(-100.5,0.5,bucket_count+1)
midpoints = 0.5 * (boundaries[0:-1]+boundaries[1:])

def discretize(value_target):
    indices = torch.bucketize(value_target, boundaries, right=False)
    return indices.clamp(0,bucket_count-1)

class ValueModel(nn.Module):
    projection_weight: Tensor
    projection_bias: Tensor
    midpoints: Tensor
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
        self.register_buffer('midpoints', midpoints)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(k) for _ in range(4)])
        self.hidden_layers = nn.ModuleList([nn.Linear(k, k) for _ in range(4)])
        self.output_layer = nn.Linear(k, bucket_count)
    def forward(self, x: Tensor)->Tensor:
        x = torch.sin(F.linear(x, self.projection_weight, self.projection_bias))
        for norm, layer in zip(self.layer_norms, self.hidden_layers):
            x = x + layer(F.silu(norm(x)))
        return self.output_layer(x)
    def __call__(self, *args, **kwds)->Tensor:
        return super().__call__(*args, **kwds)
    def get_expected_value(self, state: Tensor) -> Tensor:
        logits = self.forward(state)
        probs = torch.softmax(logits, dim=-1)
        return torch.sum(probs * self.midpoints, dim=-1)
        
value_logit_model = ValueModel()
g = torch.func.vmap(torch.func.grad(value_logit_model.get_expected_value))
state = 10*torch.rand(1,26)
expected_value = value_logit_model.get_expected_value(state)
g(state)
    
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
    
