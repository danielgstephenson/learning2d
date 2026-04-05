from torch import Tensor, tensor
import torch

def get_life(state: Tensor)->Tensor:
    bladeVector = state[:,4:6]
    distance = torch.norm(bladeVector,p=2,dim=1)
    death_range = -1 # 15
    return torch.where(distance > death_range, 1, 0)

def get_objective(state: Tensor)->Tensor:
    agentVector = state[:,0:2]
    agentDistance = torch.norm(agentVector,p=2,dim=1)
    distanceError = torch.abs(agentDistance - 40)
    life = get_life(state)
    reward = life * (1000 - distanceError)
    return reward

def get_reward(state: Tensor, outcome: Tensor)->Tensor:
    return get_objective(outcome) - get_objective(state)
