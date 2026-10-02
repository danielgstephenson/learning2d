import torch
from torch import Tensor
import torch.nn.functional as F

from value import ValueModel
from physics import Agent,World,physics_dtype,time_step,agent_radius,blade_radius

class DataGenerator:
    def __init__(self,batch_size = 1):
        self.model = ValueModel()
        self.batch_size = batch_size
        self.step_count = 10
        self.horizon = 0
        self.sample_idxs = torch.arange(self.batch_size)
        self.world = World(self.batch_size)
        self.agent0 = Agent(self.world, 0)
        self.blade0 = self.agent0.blade
        self.agent1 = Agent(self.world, 1)
        self.blade1 = self.agent1.blade
        self.state: Tensor
        self.reward: Tensor
        self.reset()

    def reset(self):
        self.world.time = torch.zeros(self.world.count,1,dtype=physics_dtype)
        n = self.batch_size
        a0p = get_random_vectors(n, 150)
        a1p = get_random_vectors(n, 150)
        b0p = a0p + get_random_vectors(n, 160)
        b1p = a1p + get_random_vectors(n, 160)
        a0v = get_random_vectors(n, 120)
        a1v = get_random_vectors(n, 120)
        b0v = get_random_vectors(n, 200)
        b1v = get_random_vectors(n, 200)
        self.agent0.position = a0p
        self.agent1.position = a1p
        self.blade0.position = b0p
        self.blade1.position = b1p
        self.agent0.velocity = a0v
        self.agent1.velocity = a1v
        self.blade0.velocity = b0v
        self.blade1.velocity = b1v
        self.agent0.alive[:] = True
        self.agent1.alive[:] = True
        self.update()

    def get_state(self, agent0: Agent, agent1: Agent)->Tensor:
        tensors = [
            agent0.velocity,
            agent1.velocity,
            agent0.blade.velocity,
            agent1.blade.velocity,
            agent0.blade.position-agent0.position,
            agent1.blade.position-agent0.position,
            agent1.position-agent0.position
        ]
        return torch.cat(tensors,dim=1)
    
    def update(self):
        self.state = self.get_state(self.agent0,self.agent1)
        hit_dist = agent_radius + blade_radius
        gap_vector0 = self.agent0.position-self.blade1.position
        gap_vector1 = self.agent1.position-self.blade0.position
        self.gap0 = norm(gap_vector0)-hit_dist
        self.gap1 = norm(gap_vector1)-hit_dist
        self.agent0.alive = self.agent0.alive & (self.gap0 > 0)
        self.agent1.alive = self.agent1.alive & (self.gap1 > 0)
        life0 = self.agent0.alive.float()
        life1 = self.agent1.alive.float()
        self.reward = life0 - life1

    def get_end_prob(self)->float:
        if self.horizon == 0: return 1
        return max(0, min(1, time_step/self.horizon))
        
    def generate(self,stage: int)->tuple[Tensor,Tensor]:
        p = self.get_end_prob()
        self.reset()
        state = self.state.clone()
        value = torch.zeros(self.batch_size,1)
        with torch.no_grad():
            for step in range(self.step_count):
                value += p*(1-p)**step*self.reward 
                if stage == 0:
                    self.agent0.action[:] = 0
                else:
                    self.agent0.action = self.model.action(self.state)
                ongoing = self.agent0.alive & self.agent1.alive
                dt = torch.where(ongoing, time_step, 0)
                self.world.step(dt)
                self.update()
            value += (1-p)**self.step_count*self.model(self.state)
            return state, value

def get_random_directions(count: int)->Tensor:
    normals = torch.randn(count, 2)
    unit = F.normalize(normals,p=2,dim=1)
    return unit

def get_random_vectors(count: int, max_scale=1.0) ->Tensor:
    directions = get_random_directions(count)
    scales = max_scale*torch.sqrt(torch.rand(count)).unsqueeze(1)
    return scales*directions

def norm(x: Tensor)->Tensor:
    return torch.norm(x,dim=1,keepdim=True)