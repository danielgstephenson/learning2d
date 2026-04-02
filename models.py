import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from generator import DataGenerator
from physics import device
import os

from reward import get_life, get_reward

class ValueModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_dim = 18
        k = 200
        self.hidden_count = 4
        self.projection_layer = nn.Linear(self.input_dim, k)
        self.hidden_layers = nn.ModuleList()
        for _ in range(self.hidden_count):
            self.hidden_layers.append(nn.Linear(k, k))
        self.output_layer = nn.Linear(k, 1)
    def forward(self, x: Tensor)->Tensor:
        x = self.projection_layer(x)
        for i in range(self.hidden_count):
            h = self.hidden_layers[i]
            x = x + F.leaky_relu(h(x),negative_slope=0.01)
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

discount = 0.9
other_noise = 0.7
def get_action_values(value_model: ValueModel, state: Tensor, outcomes: Tensor, horizon: int):
    with torch.no_grad():
        reward = get_reward(state).reshape(-1,1,1)
        life = 1 # get_life(state).reshape(-1,1,1)
        if horizon > 1:
            next_values = value_model(outcomes).reshape((-1,9,9))
        else:
            next_values = get_reward(outcomes).reshape((-1,9,9))
        values = (1-discount)*reward + discount*next_values
        values = life*values + (1-life)*reward
        row_means = torch.mean(values,2)
        row_mins = torch.amin(values,2)
        action_values = (1-other_noise)*row_mins + other_noise*row_means
    return action_values