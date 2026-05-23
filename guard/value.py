import torch
from torch import nn, Tensor
import torch.nn.functional as F

state_size = 16

class FrequencyLayer(nn.Module):
    def __init__(self, in_dim, out_dim, is_first=False, frequency=30.0):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.frequency_scale = frequency / 1.3133
        self.frequency_raw = nn.Parameter(torch.empty(out_dim).uniform_(0.5,1.5))
        self.is_first = is_first
        self.in_dim = in_dim
        self._init_weights()
    def _init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_dim, 1 / self.in_dim)
            else:
                bound = (6 / self.in_dim) ** 0.5
                self.linear.weight.uniform_(-bound, bound)
    def forward(self, x):
        frequency = F.softplus(self.frequency_raw) * self.frequency_scale
        return torch.sin(frequency * self.linear(x))

class FrequencyValueModel2(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_dim = state_size
        k = 512
        self.projection = FrequencyLayer(self.input_dim, k, is_first=True)
        self.hidden_layers = nn.ModuleList([FrequencyLayer(k, k) for _ in range(4)])
        self.output_layer = nn.Linear(k, 1)
        with torch.no_grad():
            bound = (6 / k) ** 0.5
            self.output_layer.weight.uniform_(-bound, bound)
    def forward(self, x: Tensor)->Tensor:
        x = self.projection(x)
        for layer in self.hidden_layers:
            x = x + layer(x)
        return self.output_layer(x)
    def __call__(self, *args, **kwds)->Tensor:
        return super().__call__(*args, **kwds)
    
class ValueModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_dim = state_size
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

def lp_loss(predictions: Tensor, targets: Tensor, order=3) -> Tensor:
    absolute_residuals = torch.abs(predictions - targets)
    return torch.linalg.vector_norm(absolute_residuals, ord=order)
    
