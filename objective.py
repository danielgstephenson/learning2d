from torch import Tensor, tensor
from torch.functional import _return_counts
from physics import physics_dtype
import torch

from models import ValueModel

def get_reward(state: Tensor)->Tensor:
    blade_vector = state[:,4:6]
    blade_distance = torch.norm(blade_vector,p=2,dim=1, keepdim=True)
    return blade_distance.to(physics_dtype)

discount = 0.99
other_noise = 0.1
other_passive = 0
def get_action_values(value_model: ValueModel, state: Tensor, outcomes: Tensor, horizon: int):
    with torch.no_grad():
        reward = get_reward(state).repeat_interleave(81, dim=0).reshape(-1,9,9)
        if horizon > 0:
            next_values = value_model(outcomes).reshape(-1,9,9)
            value = reward + discount*next_values
        else:
            value = reward
        row_means = torch.mean(value,2)
        row_mins = torch.amin(value,2)
        action_values = other_noise*row_means + (1-other_noise)*row_mins
        action_values = other_passive*value[:,:,0] + (1-other_passive)*action_values
    return action_values
