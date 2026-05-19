from math import pi
from numpy import where
import torch
from torch import Tensor
from torch.func import vmap, grad
import torch.nn.functional as F

from value import ValueModel
from physics import Agent, Blade, Simulation, active_action_tensor, vision_cast, physics_dtype

unit_square = torch.tensor([[-1,-1],[1,-1],[1,1],[-1,1]]).to(physics_dtype)

class DataGenerator:
    def __init__(self, value_model: ValueModel, batch_size = 3, time_step = 0.1, step_count = 50, boundary_scale = 1):
        self.value_model = value_model
        self.batch_size = batch_size
        self.step_count = step_count
        self.time_step = time_step
        self.boundary_scale = boundary_scale
        self.simulation = Simulation(batch_size, time_step)
        self.get_costate = vmap(grad(lambda x: self.value_model(x).sum()))
        self.agent0 = Agent(self.simulation, 0)
        self.blade0 = Blade(self.simulation, self.agent0)
        self.agent1 = Agent(self.simulation, 1)
        self.blade1 = Blade(self.simulation, self.agent1)
        self.radius: Tensor
        self.rotation: Tensor
        self.scale: Tensor
        self.state: Tensor
        self.costate: Tensor
        self.vgrad0: Tensor
        self.vgrad1: Tensor
        self.life0: Tensor
        self.life1: Tensor
        self.reset()

    def setup_boundary(self):
        angle = torch.rand(self.batch_size)*2*pi
        cosAngle = torch.cos(angle)
        sinAngle = torch.sin(angle)
        x_row = torch.stack((cosAngle, -sinAngle), dim=-1)
        y_row = torch.stack((sinAngle, cosAngle), dim=-1)
        self.rotation = torch.stack((x_row, y_row), dim=1).to(physics_dtype)
        self.radius = 50*(1+self.boundary_scale*torch.rand(self.batch_size,1,1))
        corners = unit_square.unsqueeze(0) * self.radius
        rotated_corners = torch.einsum('bij,bkj->bki', self.rotation, corners)
        self.simulation.boundary.setup(rotated_corners)
    
    def reset(self):
        self.setup_boundary()
        radius2d = self.radius.squeeze(-1)
        agentBound = radius2d - self.agent0.radius
        agentPosition0 = agentBound * (1 - 2 * torch.rand(self.batch_size,2))
        agentPosition1 = agentBound * (1 - 2 * torch.rand(self.batch_size,2))
        bladeBound = torch.zeros(self.batch_size, 2) + radius2d - self.blade0.radius
        bladeMax0 = torch.min(agentPosition0 + 100, +bladeBound)
        bladeMin0 = torch.max(agentPosition0 - 100, -bladeBound)
        bladeRange0 = bladeMax0 - bladeMin0
        bladePosition0 = bladeMin0 + bladeRange0 * torch.rand(self.batch_size,2)
        bladeMax1 = torch.min(agentPosition1 + 100, +bladeBound)
        bladeMin1 = torch.max(agentPosition1 - 100, -bladeBound)
        bladeRange1 = bladeMax1 - bladeMin1
        bladePosition1 = bladeMin1 + bladeRange1 * torch.rand(self.batch_size,2)
        self.agent0.position = torch.einsum('bij,bj->bi', self.rotation, agentPosition0)
        self.blade0.position = torch.einsum('bij,bj->bi', self.rotation, bladePosition0)
        self.agent1.position = torch.einsum('bij,bj->bi', self.rotation, agentPosition1)
        self.blade1.position = torch.einsum('bij,bj->bi', self.rotation, bladePosition1)
        self.agent0.velocity = get_random_vectors(self.batch_size,30)
        self.agent1.velocity = get_random_vectors(self.batch_size,30)
        self.blade0.velocity = get_random_vectors(self.batch_size,70)
        self.blade1.velocity = get_random_vectors(self.batch_size,70)

    def update(self, horizon: int):
        self.state = get_simulation_state(self.simulation)
        gap0 = torch.norm(self.agent0.position - self.blade1.position,p=2,dim=1,keepdim=True)
        gap1 = torch.norm(self.agent1.position - self.blade0.position,p=2,dim=1,keepdim=True)
        # charge_vector = F.normalize(self.agent1.position - self.agent0.position, dim=1)
        # safety_factor = 0.1 * torch.clamp((gap0 - 35) / 35, 0.0, 1.0)
        # bias_values = torch.einsum('ij,kj->ik', charge_vector * safety_factor, active_action_tensor)
        self.life0 = torch.where(gap0 > 15, 1, 0).to(physics_dtype)
        self.life1 = torch.where(gap1 > 15, 1, 0).to(physics_dtype)
        if horizon==0:
            self.costate = 0*self.state
            self.vgrad0 = +self.costate[:,[0,1]]
            self.agent0.action = torch.zeros(self.batch_size).int()
            self.agent1.action = torch.zeros(self.batch_size).int()
        else:
            self.costate = self.get_costate(self.state)
            self.vgrad0 = +self.costate[:,[0,1]]
            action_values0 = torch.einsum('ij,kj->ik',self.vgrad0,active_action_tensor)
            self.agent0.action = torch.argmax(action_values0, dim=1)+1
            self.agent1.action = torch.zeros(self.batch_size).int()

    def generate(self, horizon: int)->tuple[Tensor,...]:
        self.value_model.eval()
        p = 0.001 # Discount Rate
        c = 0.00 # Caution
        with torch.no_grad():
            self.reset()
            self.update(horizon)
            state = self.state.clone()
            life0 = self.life0.clone()
            life1 = self.life1.clone()
            reward = c * life0 + (1-c) * life0 * (1 - life1)
            value_target = reward * p
            for t in range(self.step_count):
                self.simulation.step()
                self.update(horizon)
                complete = (life0 * life1 == 0)
                life0 = torch.where(complete, life0, life0*self.life0)
                life1 = torch.where(complete, life1, life1*self.life1)
                end_prob = p * (1-p) ** (t+1)    
                reward = c * life0 + (1-c) * life0 * (1 - life1)
                value_target += reward * end_prob
            outcome = self.state.clone()
            reward = c * life0 + (1-c) * life0 * (1 - life1)
            complete = (life0*life1 == 0)
            current_value = F.sigmoid(self.value_model(outcome))
            continuation_value = reward if horizon == 0 else torch.where(complete, reward, current_value)
            continuation_prob = (1-p) ** (self.step_count+1) 
            value_target += continuation_value * continuation_prob
            return state, value_target

vision_reach = 100
def get_simulation_state(simulation: Simulation)->Tensor:
    vision = vision_cast(simulation.agents[0].position, vision_reach, simulation.boundary)
    stateTensors = [
        simulation.agents[0].velocity,
        simulation.blades[0].position - simulation.agents[0].position,
        simulation.blades[0].velocity,
        simulation.agents[1].position - simulation.agents[0].position,
        simulation.agents[1].velocity,
        simulation.blades[1].position - simulation.agents[0].position,
        simulation.blades[1].velocity,
        vision.reshape(-1,16),
    ]
    simulation_state = torch.cat(stateTensors,dim=1)
    return simulation_state

def get_random_directions(count: int)->Tensor:
    normals = torch.randn(count, 2)
    unit = F.normalize(normals,p=2,dim=1)
    return unit

def get_random_vectors(count: int, max_scale=1) ->Tensor:
    directions = get_random_directions(count)
    scales = max_scale*torch.rand(count).unsqueeze(1)
    return scales*directions

