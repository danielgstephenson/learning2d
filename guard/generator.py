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
            discount_rate=1/100,
            state_noise=3,
            action_noise=0.1,
            time_step=0.1):
        self.value_model = value_model
        self.gradient = vmap(grad(lambda x: self.value_model(x).sum()))
        self.batch_size = batch_size
        self.step_count = step_count
        self.discount_rate = discount_rate
        self.state_noise = state_noise
        self.action_noise = action_noise
        self.time_step = time_step
        self.sample_idxs = torch.arange(self.batch_size)
        self.world = World(self.batch_size, self.time_step)
        self.ring_size = 13
        self.agent0 = Agent(self.world, 0)
        self.blade0 = Blade(self.world, self.agent0)
        self.agent1 = Agent(self.world, 1)
        self.blade1 = Blade(self.world, self.agent1)
        self.world.boundary = Boundary(self.world)
        self.rotation: Tensor
        self.radius: Tensor
        self.box_offset: Tensor
        self.state: Tensor
        self.gap0: Tensor
        self.gap1: Tensor
        self.reward: Tensor
        self.reset()
    
    def setup_boundary(self):
        n = self.batch_size
        angle = torch.rand(n) * 2 * pi
        cos_angle = torch.cos(angle)
        sin_angle = torch.sin(angle)
        xs = torch.stack((cos_angle, -sin_angle), dim=-1)
        ys = torch.stack((sin_angle,  cos_angle), dim=-1)
        self.rotation = torch.stack((xs, ys), dim=1).to(physics_dtype)   # (n,2,2)
        self.radius = (40 + 100 * torch.rand(n, 1, 1)).to(physics_dtype)  # (n,1,1)
        max_offset = (self.radius.squeeze(-1) - self.ring_size).clamp(min=0)   # (n,1)
        offset_scale = torch.rand(n, 2) ** 2
        self.box_offset = max_offset * (1 - 2 * torch.rand(n, 2)) * offset_scale  # (n,2)
        corners_local = unit_square.unsqueeze(0) * self.radius + self.box_offset.unsqueeze(1) # (n,4,2)
        rotated_corners = torch.einsum('bij,bkj->bki', self.rotation, corners_local)
        self.world.boundary.setup(rotated_corners)

    def reset(self):
        self.world.time = 0
        n = self.batch_size
        self.setup_boundary()
        radiusColumn = self.radius.squeeze(-1)
        a0p_local = self.box_offset + radiusColumn * (1 - 2 * torch.rand(n, 2))
        a1p_local = self.box_offset + radiusColumn * (1 - 2 * torch.rand(n, 2))
        ring_radius = self.ring_size - self.agent1.radius
        # Oversample near ring states
        a0p_near = get_random_vectors(n, 5*ring_radius)
        a0p_local = torch.where(torch.rand(n,1) < 0.5, a0p_near, a0p_local)
        a1p_near = get_random_vectors(n, 5*ring_radius)
        a1p_local = torch.where(torch.rand(n,1) < 0.5, a1p_near, a1p_local)
        # Oversample inside ring states
        a0p_inside = get_random_vectors(n, ring_radius)
        a0p_local = torch.where(torch.rand(n,1) < 0.5, a0p_inside, a0p_local)
        a1p_inside = get_random_vectors(n, ring_radius)
        a1p_local = torch.where(torch.rand(n,1) < 0.5, a1p_inside, a1p_local)
        # Clamp to bounds
        agent_bound = radiusColumn - self.agent1.radius
        min_ap = self.box_offset - agent_bound
        max_ap = self.box_offset + agent_bound
        a0p_local = torch.clamp(a0p_local, min_ap, max_ap)
        a1p_local = torch.clamp(a1p_local, min_ap, max_ap)
        # Position Blades
        blade_bound = radiusColumn - self.blade0.radius  # (n,1)
        b0_max = torch.min(a0p_local + 65, self.box_offset + blade_bound)
        b0_min = torch.max(a0p_local - 65, self.box_offset - blade_bound)
        b0p_local = b0_min + (b0_max - b0_min) * torch.rand(n, 2)
        b1_max = torch.min(a1p_local + 65, self.box_offset + blade_bound)
        b1_min = torch.max(a1p_local - 65, self.box_offset - blade_bound)
        b1p_local = b1_min + (b1_max - b1_min) * torch.rand(n, 2)
        life0 = torch.rand(n, 1) < 0.5
        life1 = torch.rand(n, 1) < 0.5
        a0p = torch.einsum('bij,bj->bi', self.rotation, a0p_local)
        a1p = torch.einsum('bij,bj->bi', self.rotation, a1p_local)
        b0p = torch.einsum('bij,bj->bi', self.rotation, b0p_local)
        b1p = torch.einsum('bij,bj->bi', self.rotation, b1p_local)
        a0v = get_random_vectors(n, 30)
        a1v = get_random_vectors(n, 30)
        b0v = get_random_vectors(n, 45)
        b1v = get_random_vectors(n, 45)
        charge = torch.rand(self.batch_size,1)
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
        self.world.charge = charge
        self.update()

    def reset_custom(self): # Only works for batch_size = 1
        self.reset()
        r_val = (self.radius[0] - self.agent0.radius).item() * 0.9
        a0p_local = self.box_offset[0] + torch.tensor([r_val, r_val])
        a1p_local = torch.zeros(2)
        self.agent0.position[0] = torch.einsum('ij,j->i', self.rotation[0], a0p_local)
        self.agent1.position[0] = torch.einsum('ij,j->i', self.rotation[0], a1p_local)
        self.agent0.velocity[0] = torch.zeros(2)
        self.agent1.velocity[0] = torch.zeros(2)
        blade_bound = (self.radius[0] - self.blade0.radius).squeeze()
        b0_max = torch.min(a0p_local + 65, self.box_offset[0] + blade_bound)
        b0_min = torch.max(a0p_local - 65, self.box_offset[0] - blade_bound)
        b1_max = torch.min(a1p_local + 65, self.box_offset[0] + blade_bound)
        b1_min = torch.max(a1p_local - 65, self.box_offset[0] - blade_bound)
        self.blade0.position[0] = torch.einsum('ij,j->i', self.rotation[0], b0_min + (b0_max - b0_min) * torch.rand(2))
        self.blade1.position[0] = torch.einsum('ij,j->i', self.rotation[0], b1_min + (b1_max - b1_min) * torch.rand(2))
        self.agent0.alive = torch.ones_like(self.agent0.alive).bool()
        self.agent1.alive = torch.ones_like(self.agent1.alive).bool()
        self.world.charge = torch.zeros(self.world.count,1)
        self.update()

    def update(self):
        self.state = self.get_state()
        gapVector0 = self.agent0.position-self.blade1.position
        gapVector1 = self.agent1.position-self.blade0.position
        self.gap0 = torch.norm(gapVector0,dim=1,keepdim=True)-15
        self.gap1 = torch.norm(gapVector1,dim=1,keepdim=True)-15
        self.agent0.alive = self.agent0.alive & (self.gap0 > 0)
        self.agent1.alive = self.agent1.alive & (self.gap1 > 0)
        center_dist0 = torch.norm(self.agent0.position,dim=1,keepdim=True)
        center_dist1 = torch.norm(self.agent1.position,dim=1,keepdim=True)
        key_dist = self.ring_size - self.agent0.radius
        ringDist0 = center_dist0 - key_dist
        ringDist1 = center_dist1 - key_dist
        inRing1 = ringDist1 < 0
        nearRing0 = torch.sigmoid(-0.3*ringDist0)
        nearRing1 = torch.sigmoid(-0.3*ringDist1)
        charging0 = (self.agent0.alive*nearRing0).float()
        charging1 = (self.agent1.alive*nearRing1).float()
        life0 = 0.5*self.agent0.alive.float()
        life1 = 0.5*self.agent1.alive.float()
        safe0 = 0.5*life0 + 0.5*torch.sigmoid(0.3*self.gap0)
        safe1 = 0.5*life1 + 0.5*torch.sigmoid(0.3*self.gap1)
        reward0 = 0.8*charging0 + 0.2*safe0
        reward1 = 0.8*charging1 + 0.2*safe1
        self.reward = 0.5 + 0.5*reward0 - 0.5*reward1
        self.world.charging = (self.world.charge==1) | (inRing1 & self.agent1.alive)
    

    def get_state(self)->Tensor:
        tensors = [
            self.world.agents[0].velocity,
            self.world.agents[0].position,
            self.world.blades[0].velocity,
            self.world.blades[0].position,
            self.world.agents[1].velocity,
            self.world.agents[1].position,
            self.world.blades[1].velocity,
            self.world.blades[1].position,
            self.world.charge,
            self.agent0.alive.int(),
            self.agent1.alive.int(),
        ]
        origin = self.world.agents[1].position
        wallPoints = vision_cast(origin,vision_reach,self.world.boundary)
        tensors.append(wallPoints.reshape(self.world.count, 16))
        return torch.cat(tensors,dim=1)
    
    def get_action_values(self,state:Tensor)->tuple[Tensor,Tensor]:
        grad = self.gradient(state)
        vgrad0 = grad[:,[0,1]]
        vgrad1 = grad[:,[8,9]]
        action0_values = torch.einsum('ij,kj->ik',vgrad0,action_tensor)
        action1_values = torch.einsum('ij,kj->ik',vgrad1,action_tensor)
        return action0_values, action1_values

    def get_actions(self,action_values:tuple[Tensor,Tensor])->tuple[Tensor,Tensor]:
        action0_values = action_values[0]
        action1_values = action_values[1]
        action0 = torch.argmax(action0_values,dim=1,keepdim=True)
        action1 = torch.argmin(action1_values,dim=1,keepdim=True)
        random0 = torch.randint_like(action0,low=0,high=action_count)
        random1 = torch.randint_like(action1,low=0,high=action_count)
        explore0 = torch.rand_like(action0_values) < self.action_noise
        explore1 = torch.rand_like(action1_values) < self.action_noise
        action0 = torch.where(explore0, random0, action0)
        action1 = torch.where(explore1, random1, action1)
        return action0, action1

    def generate(self,stage: int)->tuple[Tensor,Tensor]:
        p = self.discount_rate
        n = self.batch_size
        k = self.step_count
        state = torch.zeros(k,n,state_size)
        reward = torch.zeros(k,n,1)
        value = torch.zeros(k,n,1)
        with torch.no_grad():
            self.reset()
            for step in range(k):
                # if step % 10 == 0: print('.', end='', flush=True)
                noisy_state = self.state.clone()
                noisy_state[:,0:17] += self.state_noise*torch.rand(n,17)
                state[step,:,:] = noisy_state
                if stage > 0:
                    action_values = self.get_action_values(noisy_state)
                    actions = self.get_actions(action_values)
                    self.agent0.action = actions[0]
                    self.agent1.action = actions[1]
                else:
                    self.agent0.action = torch.randint_like(self.agent0.action,0,action_count)
                    self.agent1.action = torch.randint_like(self.agent0.action,0,action_count)
                self.world.step()
                self.update()
                reward[step,:,:] = self.reward
            for back in range(k):
                step = k - back - 1
                if back==0:
                    logit = self.value_model(self.state)
                    continuation_value = torch.sigmoid(logit)
                else:
                    continuation_value = value[step+1,:,:]
                value[step,:,:] = p*reward[step] + (1-p)*continuation_value
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