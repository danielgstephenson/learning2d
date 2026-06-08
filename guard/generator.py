import torch
from torch import Tensor
from torch.func import vmap, grad
import torch.nn.functional as F
from math import pi

from models import ValueModel, state_size
from world import Agent,Blade,Boundary,World,physics_dtype,vision_cast,action_tensor,action_count

unit_square = torch.tensor([[-1,-1],[1,-1],[1,1],[-1,1]]).to(physics_dtype)
vision_reach = 400.0  # maximum raycast distance

class DataGenerator:
    def __init__(
            self, 
            value_model: ValueModel,
            batch_size = 1,
            step_count=10,
            discount=1/100,
            noise=0.1,
            time_step=0.1):
        self.radius = 200
        self.value_model = value_model
        self.gradient = vmap(grad(lambda x: self.value_model(x).sum()))
        self.batch_size = batch_size
        self.step_count = step_count
        self.discount = discount
        self.noise = noise
        self.time_step = time_step
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
        self.world.charge = 2*torch.rand(n,1)
        self.update()

    def reset_custom(self): # Only works for batch_size = 1
        self.reset()
        n = self.batch_size
        a0p = torch.zeros(n, 2)
        b0p = a0p + get_random_vectors(n, 70)
        self.agent0.position = a0p
        self.blade0.position = b0p
        self.agent0.alive = torch.ones_like(self.agent0.alive).bool()
        self.agent1.alive = torch.ones_like(self.agent1.alive).bool()
        self.world.charge = 0*self.world.charge
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
            self.world.charge,
        ]
        return torch.cat(tensors,dim=1)
    
    def get_vgrads(self,state:Tensor)->tuple[Tensor,Tensor]:
        grad = self.gradient(state)
        vgrad0 = grad[:,[0,1]]
        vgrad1 = grad[:,[2,3]]
        return vgrad0, vgrad1

    def get_action_values(self,vgrads:tuple[Tensor,Tensor])->tuple[Tensor,Tensor]:
        vgrad0, vgrad1 = vgrads
        action0_values = torch.einsum('ij,kj->ik',vgrad0,action_tensor)
        action1_values = torch.einsum('ij,kj->ik',vgrad1,action_tensor)
        return action0_values, action1_values

    def get_actions(self,action_values:tuple[Tensor,Tensor],noise:float)->tuple[Tensor,Tensor]:
        action0_values = action_values[0]
        action1_values = action_values[1]
        action0 = torch.argmax(action0_values,dim=1,keepdim=True)
        action1 = torch.argmin(action1_values,dim=1,keepdim=True)
        random0 = torch.randint_like(action0,low=0,high=action_count)
        random1 = torch.randint_like(action1,low=0,high=action_count)
        explore0 = torch.rand(action0.shape) < noise
        explore1 = torch.rand(action1.shape) < noise
        action0 = torch.where(explore0, random0, action0)
        action1 = torch.where(explore1, random1, action1)
        return action0, action1
    
    def update(self):
        self.world.charge.clamp_(0,1)
        self.state = self.get_state()
        gapVector0 = self.agent0.position-self.blade1.position
        gapVector1 = self.agent1.position-self.blade0.position
        self.gap0 = norm(gapVector0)-15
        self.gap1 = norm(gapVector1)-15
        self.agent0.alive = self.agent0.alive & (self.gap0 > 0)
        self.agent1.alive = self.agent1.alive & (self.gap1 > 0)
        center_dist0 = norm(self.agent0.position)
        center_dist1 = norm(self.agent1.position)
        key_dist = self.ring_size + self.agent0.radius
        in_ring0 = self.agent0.alive & (center_dist0<key_dist)
        in_ring1 = self.agent1.alive & (center_dist1<key_dist)
        full = self.world.charge == 1
        charging = full | (in_ring1 & ~in_ring0)
        self.world.d_charge = torch.where(charging, 1, -1)
        self.reward = 1 - self.world.charge

    def generate(self,stage: int)->tuple[Tensor,Tensor]:
        p = self.discount
        n = self.batch_size
        k = self.step_count
        state = torch.zeros(2*k,n,state_size)
        reward = torch.zeros(2*k,n,1)
        value = torch.zeros(2*k,n,1)
        charge = torch.zeros(2*k,n,1)
        with torch.no_grad():
            self.reset()
            for step in range(2*k):
                state[step,:,:] = self.state
                if stage > 0:
                    vgrads = self.get_vgrads(self.state)
                    action_values = self.get_action_values(vgrads)
                    actions = self.get_actions(action_values,self.noise)
                    self.agent0.action = actions[0]
                    self.agent1.action = actions[1]
                else:
                    self.agent0.action = torch.randint_like(self.agent0.action,0,action_count)
                    self.agent1.action = torch.randint_like(self.agent0.action,0,action_count)
                self.world.step()
                self.update()
                reward[step,:,:] = self.reward
                charge[step,:,:] = self.world.charge
            for back in range(2*k):
                step = 2*k - back - 1
                if back==0:
                    logit = self.value_model(self.state)
                    continuation_value = torch.sigmoid(logit)
                else:
                    continuation_value = value[step+1,:,:]
                full = charge[step,:,:] == 1
                continuation_value = torch.where(full, 0, continuation_value)
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