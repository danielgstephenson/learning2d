from numpy import where
from torch import Tensor, tensor
from torch.functional import _return_counts
from physics import physics_dtype
import torch

from models import ValueModel

def get_life(state: Tensor)->Tensor:
    blade_vector = state[:,4:6]
    blade_distance = torch.norm(blade_vector,p=2,dim=1, keepdim=True)
    return (blade_distance > 15).to(physics_dtype)

def get_quality(state: Tensor)->Tensor:
    return 100*get_life(state)

discount = 0.9
other_noise = 0.3
other_passive = 0
def get_action_values(value_model: ValueModel, state: Tensor, outcomes: Tensor, horizon: int):
    with torch.no_grad():
        life0 = get_life(state).repeat_interleave(81, dim=0).reshape(-1,9,9)
        life1 = get_life(outcomes).reshape(-1,9,9)
        life1 = life0 * life1
        reward = 100*(life1 - life0)
        if horizon > 0:
            next_values = value_model(outcomes).reshape(-1,9,9)
            next_values = torch.where(life0 > 0, next_values, 0)
            value = reward + discount*next_values
        else:
            value = reward
        row_means = torch.mean(value,2)
        row_mins = torch.amin(value,2)
        action_values = other_noise*row_means + (1-other_noise)*row_mins
        action_values = other_passive*value[:,:,0] + (1-other_passive)*action_values
    return action_values
