from math import cos, pi, sin
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from physics import Agent, Blade, Simulation, actions, physics_dtype, visionCast

class DataGenerator:
    def __init__(self, batch_size = 3, time_step = 0.1, step_count = 1):
        self.batch_size = batch_size
        self.step_count = step_count
        self.simulation = Simulation(batch_size, time_step)
        self.outcome_simulation = Simulation(81 * batch_size, time_step)
        self.agent0 = Agent(self.simulation, 0)
        self.agent1 = Agent(self.simulation, 1)
        self.blade1 = Blade(self.simulation, self.agent1)
        self.outcome_agent0 = Agent(self.outcome_simulation, 0)
        self.outcome_agent1 = Agent(self.outcome_simulation, 1)
        self.outcome_blade1 = Blade(self.outcome_simulation, self.outcome_agent1)
        self.outcome_agent0.action = actions.repeat_interleave(9, dim=0).repeat(self.batch_size)
        self.outcome_agent1.action = actions.repeat(9).repeat(self.batch_size)
        self.boundary_radius: Tensor
        self.rotation: Tensor
        self.scale: Tensor
        self.state: Tensor
        self.outcomes: Tensor
        self.reset()
        self.generate_outcomes()

    def setup_boundary(self):
        angle = torch.rand(self.batch_size)*2*pi
        cosAngle = torch.cos(angle)
        sinAngle = torch.sin(angle)
        self.rotation = torch.stack((
            torch.stack((cosAngle, -sinAngle)),
            torch.stack((sinAngle, +cosAngle))
        )).permute(2,0,1).to(physics_dtype)
        self.boundary_radius = 50*(1+3*torch.rand(self.batch_size,1))
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

    def generate_outcomes(self):
        self.outcome_agent0.position = self.agent0.position.repeat_interleave(81, dim=0)
        self.outcome_agent0.velocity = self.agent0.velocity.repeat_interleave(81, dim=0)
        self.outcome_agent1.position = self.agent1.position.repeat_interleave(81, dim=0)
        self.outcome_agent1.velocity = self.agent1.velocity.repeat_interleave(81, dim=0)
        self.outcome_blade1.position = self.blade1.position.repeat_interleave(81, dim=0)
        self.outcome_agent1.velocity = self.blade1.velocity.repeat_interleave(81, dim=0)
        for _ in range(self.step_count): self.outcome_simulation.step()
        self.outcomes = get_simulation_state(self.outcome_simulation)

def get_random_directions(count: int)->Tensor:
    normals = torch.randn((count, 2))
    unit = F.normalize(normals,p=2,dim=1)
    return unit

def get_random_vectors(count: int, max_scale=1) ->Tensor:
    directions = get_random_directions(count)
    scales = max_scale*torch.rand(count).unsqueeze(1)
    return scales*directions

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