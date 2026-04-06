from torch import Tensor, tensor
import torch

from models import ValueModel

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

discount = 0.98
other_noise = 0.5
other_passive = 0.5
def get_action_values(value_model: ValueModel, state: Tensor, outcomes: Tensor, horizon: int):
    with torch.no_grad():
        states = state.repeat_interleave(81, dim=0)
        reward = get_reward(states,outcomes).reshape(-1,9,9)
        if horizon > 1:
            life = get_life(state).reshape(-1,1,1)
            next_values = life*value_model(outcomes).reshape((-1,9,9))
            values = reward + discount*next_values
        else:
            values = reward
        row_means = torch.mean(values,2)
        row_mins = torch.amin(values,2)
        action_values = other_noise*row_means + (1-other_noise)*row_mins
        action_values = other_passive*values[:,:,0] + (1-other_passive)*action_values
    return action_values
