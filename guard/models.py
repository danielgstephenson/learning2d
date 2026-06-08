import torch
from torch import nn, Tensor
from torch.func import vmap, grad
import torch.nn.functional as F
from world import action_tensor, action_count

state_size = 19

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
        self._gradient = vmap(grad(lambda x: self.forward(x).sum()))
        self.noise = 0.0
    def forward(self, x: Tensor) -> Tensor:
        x = self.projection(x)
        for norm, layer in zip(self.layer_norms, self.hidden_layers):
            x = x + layer(F.celu(norm(x)))
        return self.output_layer(self.final_norm(x))
    def __call__(self, *args, **kwds)->Tensor:
        return super().__call__(*args, **kwds)
    def gradient(self,state:Tensor)->Tensor:
        return self._gradient(state)
    def vgrads(self,state:Tensor)->tuple[Tensor,Tensor]:
        grad = self.gradient(state)
        vgrad0 = grad[:,[0,1]]
        vgrad1 = grad[:,[2,3]]
        return vgrad0, vgrad1
    def action_values(self,state:Tensor)->tuple[Tensor,Tensor]:
        vgrad0, vgrad1 = self.vgrads(state)
        action0_values = torch.einsum('ij,kj->ik',vgrad0,action_tensor)
        action1_values = torch.einsum('ij,kj->ik',vgrad1,action_tensor)
        return action0_values, action1_values
    def actions(self,state:Tensor)->tuple[Tensor,Tensor]:
        action0_values, action1_values = self.action_values(state)
        action0 = torch.argmax(action0_values,dim=1,keepdim=True)
        action1 = torch.argmin(action1_values,dim=1,keepdim=True)
        random0 = torch.randint_like(action0,low=0,high=action_count)
        random1 = torch.randint_like(action1,low=0,high=action_count)
        explore0 = torch.rand(action0.shape) < self.noise
        explore1 = torch.rand(action1.shape) < self.noise
        action0 = torch.where(explore0, random0, action0)
        action1 = torch.where(explore1, random1, action1)
        return action0, action1
    

    
