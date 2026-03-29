from torch import Tensor
import torch

def get_reward(state: Tensor)->Tensor:
    agentVector0 = state[:,0:2]
    distance = torch.norm(agentVector0,p=2,dim=1)
    distanceError = torch.abs(distance - 40)
    life = get_life(state)
    return life * (5 - 0.02 * distanceError)

def get_life(state: Tensor)->Tensor:
    bladeVector0 = state[:,4:6]
    distance = torch.norm(bladeVector0,p=2,dim=1)
    return torch.where(distance > 15, 1, 0)
