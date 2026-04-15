from math import cos, pi, sin
import numpy as np
import torch
from torch import Tensor
from torch.func import vmap, grad
import torch.nn.functional as F


from models import ValueModel
from physics import Agent, Blade, Simulation, action_tensor, physics_dtype, visionCast
import physics

class DataGenerator:
    def __init__(self, batch_size = 3, time_step = 0.1, step_count = 20, discount = 0.99, noise = 0.1):
        self.batch_size = batch_size
        self.step_count = step_count
        self.discount = discount
        self.time_step = time_step
        self.noise = noise
        self.simulation = Simulation(batch_size, time_step)
        self.agent0 = Agent(self.simulation, 0)
        self.agent1 = Agent(self.simulation, 1)
        self.blade1 = Blade(self.simulation, self.agent1)
        self.boundary_radius: Tensor
        self.rotation: Tensor
        self.scale: Tensor
        self.state: Tensor
        self.costate: Tensor
        self.vgrad0: Tensor
        self.vgrad1: Tensor
        self.reward: Tensor
        self.reset()

    def setup_boundary(self):
        angle = torch.rand(self.batch_size)*2*pi
        cosAngle = torch.cos(angle)
        sinAngle = torch.sin(angle)
        self.rotation = torch.stack((
            torch.stack((cosAngle, -sinAngle)),
            torch.stack((sinAngle, +cosAngle))
        )).permute(2,0,1).to(physics_dtype)
        self.boundary_radius = 50*(1+1*torch.rand(self.batch_size,1))
        boundary_points = [
            self.boundary_radius*torch.tensor([[-1,-1]]).repeat(self.batch_size,1),
            self.boundary_radius*torch.tensor([[+1,-1]]).repeat(self.batch_size,1),
            self.boundary_radius*torch.tensor([[+1,+1]]).repeat(self.batch_size,1),
            self.boundary_radius*torch.tensor([[-1,+1]]).repeat(self.batch_size,1)
        ]
        for i in range(len(boundary_points)):
            point = boundary_points[i].to(physics_dtype)
            point = torch.einsum('kij,kj->ki', self.rotation, point)
            boundary_points[i] = point
        self.simulation.boundary.setup(boundary_points)
    
    def reset(self):
        self.setup_boundary()
        agentBound = self.boundary_radius - self.agent0.radius
        agentPosition0 = agentBound * (1 - 2 * torch.rand(self.batch_size,2))
        agentPosition1 = agentBound * (1 - 2 * torch.rand(self.batch_size,2))
        bladeBound = torch.zeros(self.batch_size, 2) + self.boundary_radius - self.blade1.radius
        bladeMax1 = torch.min(agentPosition1 + 100, +bladeBound)
        bladeMin1 = torch.max(agentPosition1 - 100, -bladeBound)
        bladeRange1 = bladeMax1 - bladeMin1
        bladePosition1 = bladeMin1 + bladeRange1 * torch.rand(self.batch_size,2)
        self.agent0.position = torch.einsum('kij,kj->ki', self.rotation, agentPosition0)
        self.agent1.position = torch.einsum('kij,kj->ki', self.rotation, agentPosition1)
        self.blade1.position = torch.einsum('kij,kj->ki', self.rotation, bladePosition1)
        self.agent0.velocity = get_random_vectors(self.batch_size,30)
        self.agent1.velocity = get_random_vectors(self.batch_size,30)
        self.blade1.velocity = get_random_vectors(self.batch_size,70)

    def update(self, value_model: ValueModel, horizon: int):
        self.state = get_simulation_state(self.simulation)
        blade_vector = self.blade1.position - self.agent0.position
        blade_distance = torch.norm(blade_vector,p=2,dim=1,keepdim=True)
        self.reward = torch.where(blade_distance > 15, 0, -100).to(physics_dtype)
        if horizon==0: 
            self.agent0.action = torch.zeros(self.batch_size).int()
            self.agent0.action = torch.zeros(self.batch_size).int()
            self.costate = 0*self.state
            self.vgrad0 = +self.costate[:,[8,9]]
            self.vgrad1 = -self.costate[:,[2,3]]
        else:
            get_per_sample_grad = vmap(grad(lambda x: value_model(x).sum()))
            self.costate = get_per_sample_grad(self.state)
            self.vgrad0 = +self.costate[:,[8,9]]
            self.vgrad1 = -self.costate[:,[2,3]]
            action_values0 = torch.einsum('ij,kj->ik',self.vgrad0, action_tensor)
            action_values1 = torch.einsum('ij,kj->ik',self.vgrad1,action_tensor)
            self.agent0.action = torch.argmax(action_values0, dim=1)
            self.agent1.action = torch.argmax(action_values1, dim=1)

    def generate(self, value_model: ValueModel, horizon: int)->tuple[Tensor,...]:
        with torch.no_grad():
            self.reset()
            self.update(value_model, horizon)
            state = self.state.clone()
            velocity_gradient = self.vgrad0.clone()
            interval_reward = self.reward.clone()
            for t in range(self.step_count):
                self.simulation.step()
                self.update(value_model, horizon)
                discount_factor = self.discount ** (t+1)
                interval_reward += discount_factor * torch.where(interval_reward == 0, self.reward, 0)
            outcome = self.state.clone()
            discount_factor = self.discount ** (self.step_count+1)
            continuation_value = torch.where(interval_reward==0, value_model(outcome), 0)
            value_target = interval_reward if horizon==0 else interval_reward + discount_factor*continuation_value
            return state, velocity_gradient, value_target

vision_reach = 100
def get_simulation_state(simulation: Simulation)->Tensor:
    stateTensors = [
        simulation.agents[1].position - simulation.agents[0].position,
        simulation.agents[1].velocity,
        simulation.blades[0].position - simulation.agents[0].position,
        simulation.blades[0].velocity,
        simulation.agents[0].velocity,
        visionCast(simulation.agents[0].position, vision_reach, simulation.boundary.walls),
    ]
    simulation_state = torch.cat(stateTensors,dim=1)
    return simulation_state

def get_random_directions(count: int)->Tensor:
    normals = torch.randn((count, 2))
    unit = F.normalize(normals,p=2,dim=1)
    return unit

def get_random_vectors(count: int, max_scale=1) ->Tensor:
    directions = get_random_directions(count)
    scales = max_scale*torch.rand(count).unsqueeze(1)
    return scales*directions

