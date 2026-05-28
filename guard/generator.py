import torch
from torch import Tensor
from torch.func import vmap, grad
import torch.nn.functional as F
from math import pi
from value import ValueModel
from world import Agent, Blade, Boundary, World, physics_dtype, vision_cast, active_actions

unit_square = torch.tensor([[-1,-1],[1,-1],[1,1],[-1,1]]).to(physics_dtype)
vision_reach = 400.0  # maximum raycast distance

class DataGenerator:
    def __init__(self, value_model: ValueModel, batch_size = 3, time_step=0.1):
        self.value_model = value_model        
        self.time_step = time_step
        self.batch_size = batch_size
        self.world_count = 64*self.batch_size
        self.world = World(self.world_count, self.time_step)
        self.ringSize = 13
        self.charge_interval = 4
        self.charge_step = self.world.time_step / self.charge_interval
        self.agent0 = Agent(self.world, 0)
        self.blade0 = Blade(self.world, self.agent0)
        self.agent1 = Agent(self.world, 1)
        self.blade1 = Blade(self.world, self.agent1)
        actionPairs = torch.cartesian_prod(active_actions,active_actions)
        actionMatrix = actionPairs.repeat(self.batch_size,1)
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
        self.radius = (40 + 40 * torch.rand(n, 1, 1)).to(physics_dtype)  # (n,1,1)
        max_offset = (self.radius.squeeze(-1) - self.ringSize).clamp(min=0)   # (n,1)
        offset_scale = torch.rand(n, 2) ** 2
        self.box_offset = max_offset * (1 - 2 * torch.rand(n, 2)) * offset_scale  # (n,2)
        corners_local = unit_square.unsqueeze(0) * self.radius + self.box_offset.unsqueeze(1) # (n,4,2)
        rotated_corners = torch.einsum('bij,bkj->bki', self.rotation, corners_local)
        rotated_corners = rotated_corners.repeat_interleave(64,dim=0)
        self.world.boundary.setup(rotated_corners)

    def reset(self):
        self.world.time = 0
        n = self.batch_size
        self.setup_boundary()
        radiusColumn = self.radius.squeeze(-1)
        a0p_local = self.box_offset + radiusColumn * (1 - 2 * torch.rand(n, 2))
        a1p_local = self.box_offset + radiusColumn * (1 - 2 * torch.rand(n, 2))
        ring_radius = self.ringSize - self.agent1.radius
        # Oversample corner states
        r_far = radiusColumn * 0.9  # (n, 1)
        far0_local = self.box_offset + torch.sign(torch.rand(n,2)-0.5) * r_far
        far1_local = self.box_offset + torch.sign(torch.rand(n,2)-0.5) * r_far
        use_far = torch.rand(n, 1) < 0.2
        a0p_local = torch.where(use_far, far0_local, a0p_local)
        a1p_local = torch.where(use_far, far1_local, a1p_local)
        # Oversample near charging states
        a0p_near = get_random_vectors(n, 5*ring_radius)
        a0p_local = torch.where(torch.rand(n,1) < 0.5, a0p_near, a0p_local)
        a1p_near = get_random_vectors(n, 3*ring_radius)
        a1p_local = torch.where(torch.rand(n,1) < 0.5, a1p_near, a1p_local)
        # Oversample charging states
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
        life0 = torch.rand(n, 1) < 0.9
        life1 = torch.rand(n, 1) < 0.95
        a0p = torch.einsum('bij,bj->bi', self.rotation, a0p_local)
        a1p = torch.einsum('bij,bj->bi', self.rotation, a1p_local)
        b0p = torch.einsum('bij,bj->bi', self.rotation, b0p_local)
        b1p = torch.einsum('bij,bj->bi', self.rotation, b1p_local)
        a0v = get_random_vectors(n, 30)
        a1v = get_random_vectors(n, 30)
        b0v = get_random_vectors(n, 45)
        b1v = get_random_vectors(n, 45)
        charge = torch.where(torch.rand(n,1)<0.5,0.999,torch.rand(n,1))
        self.agent0.alive = life0.repeat_interleave(64,dim=0)
        self.agent1.alive = life1.repeat_interleave(64,dim=0)
        self.agent0.position = a0p.repeat_interleave(64,dim=0)
        self.agent1.position = a1p.repeat_interleave(64,dim=0)
        self.blade0.position = b0p.repeat_interleave(64,dim=0)
        self.blade1.position = b1p.repeat_interleave(64,dim=0)
        self.agent0.velocity = a0v.repeat_interleave(64,dim=0)
        self.agent1.velocity = a1v.repeat_interleave(64,dim=0)
        self.blade0.velocity = b0v.repeat_interleave(64,dim=0)
        self.blade1.velocity = b1v.repeat_interleave(64,dim=0)
        self.world.charge = charge.repeat_interleave(64,dim=0)
        self.update()

    def reset_custom(self): # Only works for batch_size = 1
        self.reset()
        r_val = (self.radius[0] - self.agent0.radius).item() * 0.9
        a0p_local = self.box_offset[0] + torch.tensor([r_val, r_val])
        a1p_local = self.box_offset[0] + torch.tensor([-r_val, -r_val])
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
        self.world.charge[0] = torch.zeros(1)
        self.update()

    def get_simulation_state(self)->Tensor:
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

    def update(self):
        self.state = self.get_simulation_state()
        self.gap0 = torch.norm(self.agent0.position-self.blade1.position,p=2,dim=1,keepdim=True)
        self.gap1 = torch.norm(self.agent1.position-self.blade0.position,p=2,dim=1,keepdim=True)
        self.agent0.alive = self.agent0.alive & (self.gap0 > 15)
        self.agent1.alive = self.agent1.alive & (self.gap1 > 15)
        ring_dist1 = torch.norm(self.agent1.position, p=2, dim=1, keepdim=True)
        inRing1 = ring_dist1 < self.ringSize - self.agent1.radius
        charging = inRing1 & self.agent1.alive
        self.world.charge = torch.where(charging, self.world.charge + self.charge_step, 0).clamp(0, 1)
        self.reward =  1.0 - charging.float()

    def generate(self, horizon: int)->tuple[Tensor,...]:
        p = 0.01 # Discount Rate
        self.value_model.eval()
        with torch.no_grad():
            self.reset()
            state = self.state[::64].clone()
            reward = self.reward[::64].clone()
            if horizon==0:
                return state, reward
            self.world.step()
            self.update()
            outcome_values = torch.sigmoid(self.value_model(self.state))
            outcome_values = torch.where(self.agent1.alive, outcome_values, 1)
            q = outcome_values.view(self.batch_size,8,8,1)
            average_value = q.mean(dim=(1,2))
            minimax_value = q.amin(dim=2).amax(dim=1)
            noise = 0.01
            continuation_value = noise*average_value + (1-noise)*minimax_value
            target = p*reward + (1-p)*continuation_value
            return state, target

def get_random_directions(count: int)->Tensor:
    normals = torch.randn(count, 2)
    unit = F.normalize(normals,p=2,dim=1)
    return unit

def get_random_vectors(count: int, max_scale=1.0) ->Tensor:
    directions = get_random_directions(count)
    scales = max_scale*torch.rand(count).unsqueeze(1)
    return scales*directions