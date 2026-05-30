import torch
from torch import Tensor
from torch.func import vmap, grad
import torch.nn.functional as F
from torch.linalg import vector_norm
from math import pi

from models import ValueModel
from world import Agent, Blade, Boundary, World, physics_dtype, vision_cast, actions, active_actions

unit_square = torch.tensor([[-1,-1],[1,-1],[1,1],[-1,1]]).to(physics_dtype)
vision_reach = 400.0  # maximum raycast distance

class DataGenerator:
    def __init__(self, value_model: ValueModel, batch_size = 3, time_step=0.1):
        self.value_model = value_model        
        self.time_step = time_step
        self.batch_size = batch_size
        self.sample_idxs = torch.arange(self.batch_size)
        self.actions = active_actions
        self.action_count = active_actions.shape[0]
        actionPairs = torch.cartesian_prod(self.actions,self.actions)
        self.pair_count = actionPairs.shape[0]
        actionMatrix = actionPairs.repeat(self.batch_size,1)
        self.world_count = self.pair_count*self.batch_size
        self.world = World(self.world_count, self.time_step)
        self.ring_size = 13
        self.charge = 0
        self.charge_interval = 4
        self.agent0 = Agent(self.world, 0)
        self.blade0 = Blade(self.world, self.agent0)
        self.agent1 = Agent(self.world, 1)
        self.blade1 = Blade(self.world, self.agent1)
        self.agent0.action = actionMatrix[:,0]
        self.agent1.action = actionMatrix[:,1]
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
        rotated_corners = rotated_corners.repeat_interleave(self.pair_count,dim=0)
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
        self.agent0.alive = life0.repeat_interleave(self.pair_count,dim=0)
        self.agent1.alive = life1.repeat_interleave(self.pair_count,dim=0)
        self.agent0.position = a0p.repeat_interleave(self.pair_count,dim=0)
        self.agent1.position = a1p.repeat_interleave(self.pair_count,dim=0)
        self.blade0.position = b0p.repeat_interleave(self.pair_count,dim=0)
        self.blade1.position = b1p.repeat_interleave(self.pair_count,dim=0)
        self.agent0.velocity = a0v.repeat_interleave(self.pair_count,dim=0)
        self.agent1.velocity = a1v.repeat_interleave(self.pair_count,dim=0)
        self.blade0.velocity = b0v.repeat_interleave(self.pair_count,dim=0)
        self.blade1.velocity = b1v.repeat_interleave(self.pair_count,dim=0)
        self.update()

    def reset_custom(self): # Only works for batch_size = 1
        self.reset()
        r_val = (self.radius[0] - self.agent0.radius).item() * 0.9
        a0p_local = self.box_offset[0] + torch.tensor([r_val, r_val])
        # a1p_local = self.box_offset[0] + torch.tensor([-r_val, -r_val])
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
        self.update()
        
    def get_action_values(self)->Tensor:
        assert self.batch_size == 1, f'get_action_values requires batch_size=1, got {self.batch_size}'
        self.value_model.eval()
        with torch.no_grad():
            # Save world 0 state
            saved = [(c.position[0].clone(), c.velocity[0].clone()) for c in self.world.circles]
            saved_alive0 = self.agent0.alive[0].clone()
            saved_alive1 = self.agent1.alive[0].clone()
            # Broadcast world 0 to all worlds
            for c in self.world.circles:
                c.position[:] = c.position[0]
                c.velocity[:] = c.velocity[0]
            self.agent0.alive[:] = saved_alive0
            self.agent1.alive[:] = saved_alive1
            # Step and evaluate
            self.world.step()
            self.update()
            outcome_values = self.value_model(self.state)
            q = outcome_values.view(self.action_count, self.action_count)  # (agent0_actions, agent1_actions)
            # Restore world 0
            for c, (pos, vel) in zip(self.world.circles, saved):
                c.position[0] = pos
                c.velocity[0] = vel
            self.agent0.alive[0] = saved_alive0
            self.agent1.alive[0] = saved_alive1
            return q

    def get_minimax_actions(self)->tuple[int,int]:
        q = self.get_action_values()
        action0 = int(q[1:,:].amin(dim=1).argmax().item())+1
        action1 = int(q[:,1:].amax(dim=0).argmin().item())+1
        return action0, action1

    def update(self):
        self.state = self.get_state()
        gapVector0 = self.agent0.position-self.blade1.position
        gapVector1 = self.agent1.position-self.blade0.position
        self.gap0 = vector_norm(gapVector0,dim=1,keepdim=True)
        self.gap1 = vector_norm(gapVector1,dim=1,keepdim=True)
        self.agent0.alive = self.agent0.alive & (self.gap0 > 15)
        self.agent1.alive = self.agent1.alive & (self.gap1 > 15)
        center_dist0 = vector_norm(self.agent0.position,dim=1,keepdim=True)
        center_dist1 = vector_norm(self.agent1.position,dim=1,keepdim=True)
        key_dist = self.ring_size - self.agent0.radius
        ringDist0 = center_dist0 - key_dist
        ringDist1 = center_dist1 - key_dist
        far0 = self.agent0.alive * torch.sigmoid(-0.1*ringDist0)
        far1 = self.agent1.alive * torch.sigmoid(-0.1*ringDist1)
        self.reward = far1

    def get_state(self)->Tensor:
        origin = self.world.agents[1].position
        wallPoints = vision_cast(origin,vision_reach,self.world.boundary)   # (n,8,2)
        stateTensors = [
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
            wallPoints.reshape(self.world.count, 16)
        ]
        return torch.cat(stateTensors,dim=1)

    def generate(self, horizon: int)->tuple[Tensor,...]:
        p = 0.2 # Discount Rate
        n = self.batch_size
        self.value_model.eval()
        with torch.no_grad():
            self.reset()
            state = self.state[::self.pair_count].clone()
            reward = self.reward[::self.pair_count].clone()
            null_action = torch.zeros(n).long()
            if horizon==0:
                return state, reward, null_action, null_action
            self.world.step()
            self.update()
            outcome_values = torch.sigmoid(self.value_model(self.state))
            q = outcome_values.view(self.batch_size,self.action_count,self.action_count)  
            average_value = q.mean(dim=(1,2)).reshape(n,1)  
            action0_values = q.amin(dim=2)    
            action0 = action0_values.argmax(dim=1)   
            action1_values = q[self.sample_idxs,action0,:]     
            action1 = action1_values.argmin(dim=1)    
            minimax_value = q[self.sample_idxs,action0,action1].reshape(n,1) 
            noise = 0.1
            continuation_value = noise*average_value + (1-noise)*minimax_value
            value = p*reward + (1-p)*continuation_value
            return state, value, action0, action1

def get_random_directions(count: int)->Tensor:
    normals = torch.randn(count, 2)
    unit = F.normalize(normals,p=2,dim=1)
    return unit

def get_random_vectors(count: int, max_scale=1.0) ->Tensor:
    directions = get_random_directions(count)
    scales = max_scale*torch.rand(count).unsqueeze(1)
    return scales*directions