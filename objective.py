from torch import Tensor, tensor
from torch.functional import _return_counts
from physics import physics_dtype
import torch

from models import ValueModel

def get_life(state: Tensor)->Tensor:
    blade_vector = state[:,4:6]
    distance = torch.norm(blade_vector,p=2,dim=1)
    return torch.where(distance > 15, 1, 0)

def objective(state: Tensor)->Tensor:
    agent_vector = state[:,0:2]
    distance = torch.norm(agent_vector,p=2,dim=1)
    near_agent = torch.where(distance < 40, 1, 0)
    life = get_life(state)
    objective = life * (100 + 10*near_agent)
    return objective.to(physics_dtype)

def get_reward(state: Tensor, outcome: Tensor):
    start = objective(state)
    end = torch.where(start>0, objective(outcome), 0)
    return end - start

discount = 0.98
other_noise = 0.5
other_passive = 0.5
def get_action_values(value_model: ValueModel, state: Tensor, outcomes: Tensor, horizon: int):
    with torch.no_grad():
        states = state.repeat_interleave(81, dim=0)
        reward = get_reward(states, outcomes).reshape(-1,9,9)
        if horizon > 1:
            next_values = value_model(outcomes).reshape(-1,9,9)
            next_values = torch.where(reward > 0, next_values, 0)
            value = reward + discount*next_values
        else:
            value = reward
        row_means = torch.mean(value,2)
        row_mins = torch.amin(value,2)
        action_values = other_noise*row_means + (1-other_noise)*row_mins
        action_values = other_passive*value[:,:,0] + (1-other_passive)*action_values
    return action_values
