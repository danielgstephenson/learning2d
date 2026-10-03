import torch
from torch import nn, Tensor
from torch.func import vmap, grad
import torch.nn.functional as F
from physics import action_tensor, action_count, max_speed

state_size = 14
position_scale = 100.0

class ValueModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_dim = state_size
        width = 256
        layer_count = 4
        self.projection = nn.Linear(self.input_dim, width)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(layer_count)])
        self.hidden_layers = nn.ModuleList([nn.Linear(width, width) for _ in range(layer_count)])
        self.output_layer = nn.Linear(width, 1)
        self.final_norm = nn.LayerNorm(width)
        self.input_scale = nn.Buffer(torch.tensor([max_speed]*8+[position_scale]*6),persistent=False)
        self._gradient = vmap(grad(lambda x: self.forward(x).sum()))
        self.noise = 0.0
    def forward(self, x: Tensor) -> Tensor:
        x = self.projection(x / self.input_scale)
        for norm, layer in zip(self.layer_norms, self.hidden_layers):
            x = x + layer(F.celu(norm(x)))
        return self.output_layer(self.final_norm(x))
    def __call__(self, *args, **kwds)->Tensor:
        return super().__call__(*args, **kwds)
    def gradient(self,state:Tensor)->Tensor:
        return self._gradient(state)
    def vgrad(self,state:Tensor)->Tensor:
        grad = self.gradient(state)
        return grad[:,[0,1]]
    def action_values(self,state:Tensor)->Tensor:
        vgrad = self.vgrad(state)
        action0_values = torch.einsum('ij,kj->ik',vgrad,action_tensor)
        return action0_values
    def action(self,state:Tensor)->Tensor:
        action_values = self.action_values(state)
        action = torch.argmax(action_values,dim=1,keepdim=True)
        random = torch.randint_like(action,low=0,high=action_count)
        explore = torch.rand(action.shape) < self.noise
        action = torch.where(explore, random, action)
        return action
    

    
