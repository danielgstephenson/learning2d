import torch
from torch import nn, Tensor
from math import log
import torch.nn.functional as F

state_size = 35
    
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
        x = self.projection(x)
        for norm, layer in zip(self.layer_norms, self.hidden_layers):
            x = x + layer(F.elu(norm(x)))
        return self.output_layer(self.final_norm(x))
    def __call__(self, *args, **kwds)->Tensor:
        return super().__call__(*args, **kwds)
    
class ActionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_dim = state_size
        k = 256
        layer_count = 4
        self.projection = nn.Linear(self.input_dim, k)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(k) for _ in range(layer_count)])
        self.hidden_layers = nn.ModuleList([nn.Linear(k, k) for _ in range(layer_count)])
        self.output_layer = nn.Linear(k, 9)
        self.final_norm = nn.LayerNorm(k)
        self.noise = 0.1
    def forward(self, x: Tensor)->Tensor:
        x = 0.01 * self.projection(x)
        for norm, layer in zip(self.layer_norms, self.hidden_layers):
            x = x + layer(F.elu(norm(x)))
        return self.output_layer(self.final_norm(x))
    def __call__(self, *args, **kwds)->Tensor:
        return super().__call__(*args, **kwds)
    def logprobs(self, state: Tensor)->Tensor:
        logits = self.forward(state)
        model_log_probs = log(1-self.noise) + torch.log_softmax(logits,dim=1)
        uniform_log_probs = torch.full_like(model_log_probs, log(self.noise / 9))
        log_probs = torch.logaddexp(model_log_probs, uniform_log_probs)
        return log_probs
    def action(self, state: Tensor)->Tensor:
        log_probs = self.logprobs(state)
        return torch.multinomial(log_probs.exp(),num_samples=1,replacement=True)

    
