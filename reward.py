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
    vision = state[:,-8:]
    minVision = torch.amin(vision,dim=1)
    visionCost = 500 / (minVision + 5)
    life = get_life(state)
    reward = life * (1000 - distanceError - visionCost)
    return reward
