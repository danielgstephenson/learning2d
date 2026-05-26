import torch
from torch import Tensor
from torch.func import vmap, grad
import torch.nn.functional as F
from math import pi
from value import ValueModel, state_size
from guard.world import Agent, Blade, Boundary, World, active_action_tensor, physics_dtype, vision_cast

unit_square = torch.tensor([[-1,-1],[1,-1],[1,1],[-1,1]]).to(physics_dtype)
vision_reach = 400.0  # maximum raycast distance

class DataGenerator:
    def __init__(self, value_model: ValueModel, world_count = 3, step_count=10, time_step=0.1):
        self.value_model = value_model
        self.get_costate = vmap(grad(lambda x: self.value_model(x).sum()))
        self.world_count = world_count
        self.step_count = step_count
        self.time_step = time_step
        self.ringSize = 13
        self.chargeTarget = 4
        self.world = World(world_count, self.time_step)
        self.agent0 = Agent(self.world, 0)
        self.blade0 = Blade(self.world, self.agent0)
        self.agent1 = Agent(self.world, 1)
        self.blade1 = Blade(self.world, self.agent1)
        self.world.boundary = Boundary(self.world)
        self.rotation: Tensor
        self.radius: Tensor
        self.box_offset: Tensor
        self.state: Tensor
        self.costate: Tensor
        self.vgrad0: Tensor
        self.vgrad1: Tensor
        self.gap0: Tensor
        self.gap1: Tensor
        self.life0: Tensor
        self.life1: Tensor
        self.reward: Tensor
        self.reset()
    
    def setup_boundary(self):
        n = self.world_count
        angle = torch.rand(n) * 2 * pi
        cos_angle = torch.cos(angle)
        sin_angle = torch.sin(angle)
        xs = torch.stack((cos_angle, -sin_angle), dim=-1)
        ys = torch.stack((sin_angle,  cos_angle), dim=-1)
        self.rotation = torch.stack((xs, ys), dim=1).to(physics_dtype)   # (n,2,2)
        self.radius = (40 + 160 * torch.rand(n, 1, 1)).to(physics_dtype)  # (n,1,1)
        max_offset = (self.radius.squeeze(-1) - self.ringSize).clamp(min=0)   # (n,1)
        offset_scale = torch.rand(n, 2) ** 2
        self.box_offset = max_offset * (1 - 2 * torch.rand(n, 2)) * offset_scale  # (n,2)
        corners_local = unit_square.unsqueeze(0) * self.radius + self.box_offset.unsqueeze(1) # (n,4,2)
        rotated_corners = torch.einsum('bij,bkj->bki', self.rotation, corners_local)
        self.world.boundary.setup(rotated_corners)

    def reset(self):
        self.world.time = 0
        n = self.world_count
        self.setup_boundary()
        radiusColumn = self.radius.squeeze(-1)
        a0p_local = self.box_offset + (radiusColumn - self.agent0.radius) * (1 - 2 * torch.rand(n, 2))
        a1p_local = self.box_offset + (radiusColumn - self.agent1.radius) * (1 - 2 * torch.rand(n, 2))
        blade_bound = radiusColumn - self.blade0.radius  # (n,1)
        b0_max = torch.min(a0p_local + 65, self.box_offset + blade_bound)
        b0_min = torch.max(a0p_local - 65, self.box_offset - blade_bound)
        b0_max = torch.min(a0p_local + 65, self.box_offset + blade_bound)
        b0_min = torch.max(a0p_local - 65, self.box_offset - blade_bound)
        b0p_local = b0_min + (b0_max - b0_min) * torch.rand(n, 2)
        b1_max = torch.min(a1p_local + 65, self.box_offset + blade_bound)
        b1_min = torch.max(a1p_local - 65, self.box_offset - blade_bound)
        b1p_local = b1_min + (b1_max - b1_min) * torch.rand(n, 2)
        self.agent0.position = torch.einsum('bij,bj->bi', self.rotation, a0p_local)
        self.agent1.position = torch.einsum('bij,bj->bi', self.rotation, a1p_local)
        self.blade0.position = torch.einsum('bij,bj->bi', self.rotation, b0p_local)
        self.blade1.position = torch.einsum('bij,bj->bi', self.rotation, b1p_local)
        self.agent0.velocity = get_random_vectors(n, 30)
        self.agent1.velocity = get_random_vectors(n, 30)
        self.blade0.velocity = get_random_vectors(n, 45)
        self.blade1.velocity = get_random_vectors(n, 45)
        self.world.complete = torch.zeros(n, 1).bool()
        self.update()

    def update(self):
        self.state = get_simulation_state(self.world)
        self.gap0 = torch.norm(self.agent0.position-self.blade1.position,p=2,dim=1,keepdim=True)
        self.gap1 = torch.norm(self.agent1.position-self.blade0.position,p=2,dim=1,keepdim=True)
        self.life0 = torch.where(self.gap0 > 15, 1, 0).to(physics_dtype)
        self.life1 = torch.where(self.gap1 > 15, 1, 0).to(physics_dtype)
        centerDistance0 = torch.norm(self.agent0.position, p=2, dim=1, keepdim=True)
        centerDistance1 = torch.norm(self.agent1.position, p=2, dim=1, keepdim=True)
        charging = centerDistance1 < self.ringSize - self.agent1.radius
        self.world.charge = torch.where(charging, self.world.charge + self.world.time_step, 0)
        fullCharge = self.world.charge > self.chargeTarget
        victory = self.life1 == 0
        defeat = fullCharge | (self.life0 == 0)
        self.world.complete = victory | defeat
        ongoing = ~self.world.complete
        ringReward = centerDistance1 - centerDistance0
        completeReward = 100 * (victory - defeat)
        self.reward = torch.where(ongoing, ringReward, completeReward)

    def act(self, horizon: int):
        if horizon==0:
            self.costate = 0*self.state
            self.vgrad0 = +self.costate[:,[0,1]]
            self.agent0.action = torch.zeros(self.world_count).int()
            self.agent1.action = torch.zeros(self.world_count).int()
        else:
            self.costate = self.get_costate(self.state)
            self.vgrad0 = +self.costate[:,[0,1]]
            self.vgrad1 = -self.costate[:,[8,9]]
            action_values0 = torch.einsum('ij,kj->ik',self.vgrad0,active_action_tensor)
            action_values1 = torch.einsum('ij,kj->ik',self.vgrad1,active_action_tensor)
            self.agent0.action = torch.argmax(action_values0, dim=1)+1
            self.agent1.action = torch.argmax(action_values1, dim=1)+1

    def generate(self, horizon: int)->tuple[Tensor,...]:
        self.value_model.eval()
        with torch.no_grad():
            self.reset()
            state = self.state.clone()
            target = torch.zeros((self.world_count,1))
            p = 0.002 # Discount Rate
            for t in range(self.step_count):
                self.act(horizon)
                self.world.step()
                self.update()
                end_prob = p * (1 - p) ** t
                target += end_prob * self.reward
            bootstrap_estimate = self.reward if horizon == 0 else self.value_model(self.state)
            continuation_value = torch.where(self.world.complete, self.reward, bootstrap_estimate) 
            continuation_prob = (1-p) ** self.step_count
            target += continuation_prob * continuation_value
            return state, target

def get_simulation_state(simulation: World)->Tensor:
    wallPoints = vision_cast(simulation.agents[1].position,vision_reach,simulation.boundary)   # (n,8,2)
    stateTensors = [
        simulation.agents[0].velocity,
        simulation.agents[0].position,
        simulation.blades[0].velocity,
        simulation.blades[0].position,
        simulation.agents[1].velocity,
        simulation.agents[1].position,
        simulation.blades[1].velocity,
        simulation.blades[1].position,
        wallPoints.reshape(simulation.count, 16)
    ]
    return torch.cat(stateTensors,dim=1)

def get_random_directions(count: int)->Tensor:
    normals = torch.randn(count, 2)
    unit = F.normalize(normals,p=2,dim=1)
    return unit

def get_random_vectors(count: int, max_scale=1.0) ->Tensor:
    directions = get_random_directions(count)
    scales = max_scale*torch.rand(count).unsqueeze(1)
    return scales*directions