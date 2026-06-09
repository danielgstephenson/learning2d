import torch
from torch import Tensor
import torch.nn.functional as F

from models import ValueModel, state_size
from world import Agent,Blade,Boundary,World,physics_dtype,vision_cast,action_tensor,action_count

unit_square = torch.tensor([[-1,-1],[1,-1],[1,1],[-1,1]]).to(physics_dtype)
vision_reach = 400.0  # maximum raycast distance

class DataGenerator:
    def __init__(self,batch_size = 1,time_step=0.02):
        self.radius = 200
        self.model = ValueModel()
        self.batch_size = batch_size
        self.time_step = time_step
        self.step_count = 10
        self.discount = 1/2
        self.sample_idxs = torch.arange(self.batch_size)
        self.world = World(self.batch_size, self.time_step)
        self.ring_size = 13
        self.agent0 = Agent(self.world, 0)
        self.blade0 = Blade(self.world, self.agent0)
        self.agent1 = Agent(self.world, 1)
        self.blade1 = Blade(self.world, self.agent1)
        self.box_offset: Tensor
        self.state: Tensor
        self.gap0: Tensor
        self.gap1: Tensor
        self.reward: Tensor
        self.reset()

    def reset(self):
        self.world.time = 0
        n = self.batch_size
        a0p = get_random_vectors(n, self.radius)
        a1p = get_random_vectors(n, self.radius)
        ring_radius = self.ring_size - self.agent1.radius
        # Oversample near ring states
        a0p_near = get_random_vectors(n, 50)
        a0p = torch.where(torch.rand(n,1) < 0.5, a0p_near, a0p)
        a1p_near = get_random_vectors(n, 50)
        a1p = torch.where(torch.rand(n,1) < 0.5, a1p_near, a1p)
        # Oversample inside ring states
        a0p_inside = get_random_vectors(n, ring_radius)
        a0p = torch.where(torch.rand(n,1) < 0.5, a0p_inside, a0p)
        a1p_inside = get_random_vectors(n, ring_radius)
        a1p = torch.where(torch.rand(n,1) < 0.5, a1p_inside, a1p)
        # Position Blades
        b0p = a0p + get_random_vectors(n, 70)
        b1p = a1p + get_random_vectors(n, 70)
        life0 = torch.rand(n, 1) < 0.5
        life1 = torch.rand(n, 1) < 0.5
        a0v = get_random_vectors(n, 30)
        a1v = get_random_vectors(n, 30)
        b0v = get_random_vectors(n, 50)
        b1v = get_random_vectors(n, 50)
        self.agent0.alive = life0
        self.agent1.alive = life1
        self.agent0.position = a0p
        self.agent1.position = a1p
        self.blade0.position = b0p
        self.blade1.position = b1p
        self.agent0.velocity = a0v
        self.agent1.velocity = a1v
        self.blade0.velocity = b0v
        self.blade1.velocity = b1v
        self.update()

    def reset_custom(self): # Only works for batch_size = 1
        self.reset()
        n = self.batch_size
        a0p = torch.zeros(n, 2)
        b0p = a0p + get_random_vectors(n, 20)
        self.agent0.position = a0p
        self.blade0.position = b0p
        self.agent0.alive = torch.ones_like(self.agent0.alive).bool()
        self.agent1.alive = torch.ones_like(self.agent1.alive).bool()
        self.update()

    def get_state(self)->Tensor:
        tensors = [
            self.world.agents[0].velocity,
            self.world.agents[1].velocity,
            self.world.blades[0].velocity,
            self.world.blades[1].velocity,
            self.world.agents[0].position,
            self.world.agents[1].position,
            self.world.blades[0].position,
            self.world.blades[1].position,
            self.agent0.alive.int(),
            self.agent1.alive.int(),
        ]
        return torch.cat(tensors,dim=1)
    
    def update(self):
        self.state = self.get_state()
        gapVector0 = self.agent0.position-self.blade1.position
        gapVector1 = self.agent1.position-self.blade0.position
        self.gap0 = norm(gapVector0)-15
        self.gap1 = norm(gapVector1)-15
        self.agent0.alive = self.agent0.alive & (self.gap0 > 0)
        self.agent1.alive = self.agent1.alive & (self.gap1 > 0)
        life1 = self.agent0.alive.float()
        dist0 = norm(self.agent0.position)
        dist1 = norm(self.agent1.position)
        d_dist = dist0-dist1
        self.reward = 1.0 - life1*torch.sigmoid(0.05*unguarded)

    def generate(self,stage: int)->tuple[Tensor,Tensor]:
        p = self.discount
        n = self.batch_size
        k = self.step_count
        state = torch.zeros(2*k,n,state_size)
        reward = torch.zeros(2*k,n,1)
        value = torch.zeros(2*k,n,1)
        with torch.no_grad():
            self.reset()
            for step in range(2*k):
                state[step,:,:] = self.state
                if stage == 0:
                    self.agent0.action = torch.randint_like(self.agent0.action,0,action_count)
                    self.agent1.action = torch.randint_like(self.agent1.action,0,action_count)
                else:
                    actions = self.model.actions(self.state)
                    self.agent0.action = actions[0]
                    self.agent1.action = actions[1]
                self.world.step()
                self.update()
                reward[step,:,:] = self.reward
            for back in range(2*k):
                step = 2*k - back - 1
                if back==0:
                    logit = self.model(self.state)
                    continuation_value = torch.sigmoid(logit)
                else:
                    continuation_value = value[step+1,:,:]
                value[step,:,:] = p*reward[step] + (1-p)*continuation_value
            state = state[:k,...]
            value = value[:k,...]
            state = state.reshape(k*n,state_size) 
            value = value.reshape(k*n,1)
            return state, value

def get_random_directions(count: int)->Tensor:
    normals = torch.randn(count, 2)
    unit = F.normalize(normals,p=2,dim=1)
    return unit

def get_random_vectors(count: int, max_scale=1.0) ->Tensor:
    directions = get_random_directions(count)
    scales = max_scale*torch.rand(count).unsqueeze(1)
    return scales*directions

def norm(x: Tensor)->Tensor:
    return torch.norm(x,dim=1,keepdim=True)