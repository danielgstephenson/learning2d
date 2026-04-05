from torch import Tensor
import torch

def get_life(state: Tensor)->Tensor:
    bladeVector = state[:,4:6]
    distance = torch.norm(bladeVector,p=2,dim=1)
    return torch.where(distance > 15, 1, 0)

def get_reward(state: Tensor)->Tensor:
    agentVector = state[:,0:2]
    agentDistance = torch.norm(agentVector,p=2,dim=1)
    distanceError = torch.abs(agentDistance - 40)
    velocity = state[:,8:10]
    speed = torch.norm(velocity,p=2,dim=1)
    move = torch.where(speed < 10, speed, 10)
    life = 1 # get_life(state)
    reward = life * (1000 + 10*move - distanceError)
    return reward
