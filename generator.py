import torch
from torch import Tensor
import torch.nn.functional as F

from value import ValueModel
from physics import Agent, Boundary,World,physics_dtype,time_step,agent_radius,blade_radius, vision_cast

class DataGenerator:
    def __init__(self,batch_size = 1):
        self.model = ValueModel()
        self.batch_size = batch_size
        self.step_count = 10
        self.horizon = 0.0
        self.survive_weight = 1
        self.attack_weight = 0
        self.dist_weight = 0
        self.velocity_weight = 1e-3
        self.sample_idxs = torch.arange(self.batch_size)
        self.world = World(self.batch_size)
        self.agent0 = Agent(self.world, 0)
        self.blade0 = self.agent0.blade
        self.agent1 = Agent(self.world, 1)
        self.blade1 = self.agent1.blade
        self.boundary = Boundary(self.world)
        self.radius: Tensor
        self.rotation: Tensor
        self.state: Tensor
        self.reward: Tensor
        self.ongoing: Tensor
        self.target_velocity: Tensor
        self.reset()

    def reset_boundary(self):
        n = self.batch_size
        unit_corners = torch.tensor([[-1,1],[1,1],[1,-1],[-1,-1]])
        self.radius = torch.abs(200*torch.randn(n,1)) + 100
        angle = torch.pi * torch.rand(n)
        C, S = torch.cos(angle), torch.sin(angle)
        self.rotation = torch.stack([torch.stack([C,-S],dim=-1), torch.stack([S, C], dim=-1)], dim=-2)
        corners = (unit_corners * self.radius[:,None]).to(physics_dtype)
        corners = torch.einsum('nij,nkj->nki', self.rotation, corners)
        self.boundary.setup(corners)

    def reset(self):
        self.world.time = torch.zeros(self.world.count,1,dtype=physics_dtype)
        self.reset_boundary()
        n = self.batch_size
        A = self.radius - agent_radius
        B = self.radius - blade_radius
        a0p = 2*A*torch.rand(n, 2) - A
        a1p = 2*A*torch.rand(n, 2) - A
        b0p = (a0p + get_random_vectors(n, 160)).clamp(-B,B)
        b1p = (a1p + get_random_vectors(n, 160)).clamp(-B,B)
        a0p = torch.einsum('nij,nj->ni', self.rotation, a0p)
        a1p = torch.einsum('nij,nj->ni', self.rotation, a1p)
        b0p = torch.einsum('nij,nj->ni', self.rotation, b0p)
        b1p = torch.einsum('nij,nj->ni', self.rotation, b1p)
        self.target_velocity = get_random_vectors(n, 120)
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
            self.target_velocity,
            agent0.blade.position-agent0.position,
            agent1.blade.position-agent0.position,
            agent1.position-agent0.position,
            vision_cast(agent0)
        ]
        return torch.cat(tensors,dim=1)
    
    def update(self):
        self.state = self.get_state(self.agent0,self.agent1)
        self.ongoing = self.agent0.alive # & self.agent1.alive
        hit_dist = agent_radius + blade_radius
        gap_vector0 = self.agent0.position-self.blade1.position
        gap_vector1 = self.agent1.position-self.blade0.position
        self.gap0 = norm(gap_vector0)-hit_dist
        self.gap1 = norm(gap_vector1)-hit_dist
        self.agent0.alive = self.agent0.alive & (self.gap0 > 0)
        self.agent1.alive = self.agent1.alive & (self.gap1 > 0)
        life0 = self.agent0.alive.float()
        life1 = self.agent1.alive.float()
        dist = norm(self.agent0.position-self.agent1.position)
        velocity_error = norm(self.agent0.velocity-self.target_velocity)
        self.reward = \
            self.survive_weight*life0 \
            - self.attack_weight*life1 \
            - self.dist_weight*dist \
            - self.velocity_weight*velocity_error


    def get_end_prob(self)->float:
        if self.horizon == 0: return 1
        return max(0, min(1, time_step/self.horizon))
        
    def generate(self,horizon: float)->tuple[Tensor,Tensor]:
        p = self.get_end_prob()
        self.reset()
        state = self.state.clone()
        value = torch.zeros(self.batch_size,1)
        with torch.no_grad():
            for step in range(self.step_count):
                value += p*(1-p)**step*self.reward 
                if horizon == 0:
                    self.agent0.action[:] = 0
                else:
                    self.agent0.action = self.model.action(self.state)
                dt = torch.where(self.ongoing, time_step, 0)
                self.world.step(dt)
                self.update()
            estimate = self.model(self.state)
            continuation_value = torch.where(self.ongoing,estimate,self.reward)
            value += (1-p)**self.step_count*continuation_value
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