import torch
from torch import Tensor
from torch.func import vmap, grad
import torch.nn.functional as F
from math import pi

from models import ActionModel, ValueModel, state_size
from world import Agent, Blade, Boundary, World, physics_dtype, vision_cast

unit_square = torch.tensor([[-1,-1],[1,-1],[1,1],[-1,1]]).to(physics_dtype)
vision_reach = 400.0  # maximum raycast distance

class DataGenerator:
    def __init__(
            self, 
            value_model: ValueModel,
            action0_model: ActionModel,
            action1_model: ActionModel,
            batch_size = 1,
            step_count=10,
            time_step=0.1):
        self.value_model = value_model
        self.action0_model = action0_model
        self.action1_model = action1_model
        self.step_count = step_count
        self.time_step = time_step
        self.batch_size = batch_size
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
        # Oversample corner states
        r_far = radiusColumn * 0.9  # (n, 1)
        far0_local = self.box_offset + torch.sign(torch.rand(n,2)-0.5) * r_far
        far1_local = self.box_offset + torch.sign(torch.rand(n,2)-0.5) * r_far
        use_far = torch.rand(n, 1) < 0.2
        a0p_local = torch.where(use_far, far0_local, a0p_local)
        a1p_local = torch.where(use_far, far1_local, a1p_local)
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
        self.state = self.get_state_vector()
        gapVector0 = self.agent0.position-self.blade1.position
        gapVector1 = self.agent1.position-self.blade0.position
        self.gap0 = torch.norm(gapVector0,dim=1,keepdim=True)
        self.gap1 = torch.norm(gapVector1,dim=1,keepdim=True)
        self.agent0.alive = self.agent0.alive & (self.gap0 > 15)
        self.agent1.alive = self.agent1.alive & (self.gap1 > 15)
        center_dist1 = torch.norm(self.agent1.position,dim=1,keepdim=True)
        key_dist = self.ring_size - self.agent0.radius
        ringDist1 = center_dist1 - key_dist
        inRing1 = ringDist1 < 0
        self.reward = 1 - (self.agent1.alive*inRing1).float()
        self.world.charging = (self.world.charge==1) | (inRing1 & self.agent1.alive)

    def get_state(self)->list[Tensor]:
        state = [
            self.world.agents[0].velocity,
            self.world.agents[0].position,
            self.world.blades[0].velocity,
            self.world.blades[0].position,
            self.world.agents[1].velocity,
            self.world.agents[1].position,
            self.world.blades[1].velocity,
            self.world.blades[1].position,
            self.agent0.alive.int(),
            self.agent1.alive.int(),
            self.world.charge
        ]
        return state
    
    def load_state(self,state: list[Tensor]):
        self.world.agents[0].velocity = state[0]
        self.world.agents[0].position = state[1]
        self.world.blades[0].velocity = state[2]
        self.world.blades[0].position = state[3]
        self.world.agents[1].velocity = state[4]
        self.world.agents[1].position = state[5]
        self.world.blades[1].velocity = state[6]
        self.world.blades[1].position = state[7]
        self.agent0.alive = (state[8]==1)
        self.agent1.alive = (state[9]==1)
        self.world.charge = state[10]
        self.update()

    def get_state_vector(self)->Tensor:
        state = self.get_state()
        origin = self.world.agents[1].position
        wallPoints = vision_cast(origin,vision_reach,self.world.boundary)   # (n,8,2)
        state.append(wallPoints.reshape(self.world.count, 16))
        return torch.cat(state,dim=1)

    def generate(self)->tuple[Tensor,...]:
        p = 1/100 # Discount Rate
        n = self.batch_size
        k = self.step_count
        state = torch.zeros(2*k,n,state_size)
        action0 = torch.zeros(2*k,n,1).int()
        action1 = torch.zeros(2*k,n,1).int()
        reward = torch.zeros(2*k,n,1)
        value_prediction = torch.zeros(2*k,n,1)
        value = torch.zeros(2*k,n,1)
        with torch.no_grad():
            self.reset()
            for step in range(2*k):
                if step % 10 == 0:
                    print('.', end='', flush=True)
                state[step,:,:] = self.state
                value_prediction[step,:,:] = self.value_model(self.state)
                a0 = self.action0_model.action(self.state)
                a1 = self.action1_model.action(self.state)
                self.agent0.action = a0
                self.agent1.action = a1
                action0[step,:] = a0
                action1[step,:] = a1
                self.world.step()
                self.update()
                reward[step,:,:] = self.reward
            for back in range(2*k):
                step = 2*k - back - 1
                if back==0:
                    future = self.value_model(self.state)
                else:
                    future = value[step+1,:,:]
                value[step,:,:] = p*reward[step] + (1-p)*future
            state = state[:k].reshape(k*n,state_size) 
            value_prediction = value_prediction[:k].reshape(k*n,1)
            value = value[:k].reshape(k*n,1)
            advantage = value - value_prediction
            action0 = action0[:k].reshape(k*n,1) 
            action1 = action1[:k].reshape(k*n,1)
            print()
            return state,value,action0,action1,advantage

def get_random_directions(count: int)->Tensor:
    normals = torch.randn(count, 2)
    unit = F.normalize(normals,p=2,dim=1)
    return unit

def get_random_vectors(count: int, max_scale=1.0) ->Tensor:
    directions = get_random_directions(count)
    scales = max_scale*torch.rand(count).unsqueeze(1)
    return scales*directions