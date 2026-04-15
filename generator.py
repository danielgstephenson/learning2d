from math import cos, pi, sin
import numpy as np
import torch
from torch import Tensor
from torch.func import vmap, grad
import torch.nn.functional as F


from models import ValueModel
from physics import Agent, Blade, Simulation, action_tensor, physics_dtype, visionCast

class DataGenerator:
    def __init__(self, batch_size = 3, time_step = 0.1, step_count = 20, discount = 0.99, noise = 0.1):
        self.batch_size = batch_size
        self.step_count = step_count
        self.discount = discount
        self.time_step = time_step
        self.noise = noise
        self.simulation = Simulation(batch_size, time_step)
        self.outcome_simulation = Simulation(81 * batch_size, time_step)
        self.agent0 = Agent(self.simulation, 0)
        self.agent1 = Agent(self.simulation, 1)
        self.blade1 = Blade(self.simulation, self.agent1)
        self.boundary_radius: Tensor
        self.rotation: Tensor
        self.scale: Tensor
        self.state: Tensor
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
        outcome_boundary_points = [point.repeat_interleave(81, dim=0) for point in boundary_points]
        self.outcome_simulation.boundary.setup(outcome_boundary_points)
    
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
        self.state = get_simulation_state(self.simulation)

    def get_reward(self)->Tensor:
        blade_vector = self.blade1.position - self.agent0.position
        blade_distance = torch.norm(blade_vector,p=2,dim=1,keepdim=True)
        reward = torch.where(blade_distance > 15, 0, -100).to(physics_dtype)
        return reward
    
    # It might be better to use torch.func.grad to get the gradient of the value function
    def get_action_values(self, agent_index: int, value_model: ValueModel)->Tensor:
        columns = [8,9] if agent_index == 0 else [2,3]
        start_values = value_model(self.state).repeat_interleave(9,dim=0)
        outcomes = self.state.repeat_interleave(9,dim=0)
        action_vectors = action_tensor.repeat(self.batch_size, 1)
        dt = 0.1*self.time_step
        outcomes[:,columns] += dt*action_vectors
        outcome_values = value_model(outcomes)
        gain0 = ((outcome_values - start_values) / dt).reshape(self.batch_size, 9)
        sign = 1 if agent_index == 0 else -1
        return sign * gain0
    
    def get_action(self, agent_index: int, value_model: ValueModel, horizon: int)->Tensor:
        random_action = torch.randint(high=9,size=(self.batch_size,))
        if horizon==0: return random_action
        action_values = self.get_action_values(agent_index, value_model)
        best_action = torch.argmax(action_values, dim=1)
        runif = torch.rand(self.batch_size)
        action = torch.where(runif < self.noise, random_action, best_action)
        return action

    def generate(self, horizon: int, old_value_model: ValueModel, value_model: ValueModel)->tuple[Tensor,...]:
        with torch.no_grad():
            self.reset()
            action_values = self.get_action_values(0,value_model)
            start = self.state.clone()
            reward = self.get_reward()
            
            def single_sample_value(x):
                return value_model(x).sum()
            per_sample_grad = vmap(grad(single_sample_value))
            gradient = per_sample_grad(start)
            print('gradient.shape',gradient.shape)

            for t in range(self.step_count):
                self.agent0.action = self.get_action(0,old_value_model,horizon)
                self.agent1.action = self.get_action(1,old_value_model,horizon)
                self.simulation.step()
                discount_factor = self.discount ** (t+1)
                reward += discount_factor * torch.where(reward == 0, self.get_reward(), 0)
            outcome = get_simulation_state(self.simulation)
            discount_factor = self.discount ** (self.step_count+1)
            continuation_value = torch.where(reward==0, old_value_model(outcome), 0)
            value_target = reward if horizon==0 else reward + discount_factor*continuation_value
            return start, value_target, action_values

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

