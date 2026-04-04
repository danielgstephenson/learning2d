from math import cos, pi, sin
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from physics import Agent, Blade, Boundary, Simulation, actions, visionCast,floatType

visionReach = 100
def get_simulation_state(simulation: Simulation)->Tensor:
    stateTensors = [
        simulation.agents[1].position - simulation.agents[0].position,
        simulation.agents[1].velocity,
        simulation.blades[0].position - simulation.agents[0].position,
        simulation.blades[0].velocity,
        simulation.agents[0].velocity,
        visionCast(simulation.agents[0].position, visionReach, simulation.boundary.walls),
    ]
    simulation_state = torch.cat(stateTensors,dim=1)
    return simulation_state

class DataGenerator:
    def __init__(self, batch_size = 3, timeStep = 0.1):
        self.batch_size = batch_size
        self.start_simulation = Simulation(batch_size, timeStep)
        self.outcome_simulation = Simulation(81 * batch_size, timeStep)
        self.boundarySize = 50
        self.start_agent0 = Agent(self.start_simulation, 0)
        self.start_agent1 = Agent(self.start_simulation, 1)
        self.start_blade1 = Blade(self.start_simulation, self.start_agent1)
        self.outcome_agent0 = Agent(self.outcome_simulation, 0)
        self.outcome_agent1 = Agent(self.outcome_simulation, 1)
        self.outcome_blade1 = Blade(self.outcome_simulation, self.outcome_agent1)
        self.outcome_agent0.action = actions.repeat_interleave(9, dim=0).repeat(self.batch_size)
        self.outcome_agent1.action = actions.repeat(9).repeat(self.batch_size)
        self.rotation: Tensor
        self.state: Tensor
        self.outcomes: Tensor
        self.reset()
        self.generate_outcomes()

    def setup_boundary(self):
        angle = np.random.rand()*2*pi
        self.rotation = torch.tensor([
            [+cos(angle), -sin(angle)],
            [+sin(angle), +cos(angle)]
        ])
        boundaryPoints = torch.tensor([
            [-self.boundarySize,-self.boundarySize],
            [+self.boundarySize,-self.boundarySize],
            [+self.boundarySize,+self.boundarySize],
            [-self.boundarySize,+self.boundarySize]
        ],dtype=floatType)
        boundaryPoints = torch.einsum('ij,kj->ki', self.rotation, boundaryPoints)
        self.outcome_simulation.boundary.setup(boundaryPoints)
        self.start_simulation.boundary.setup(boundaryPoints)
    
    def reset(self):
        self.setup_boundary()
        agentBound = self.boundarySize - self.start_agent0.radius
        agentPosition0 = agentBound * (1 - 2 * torch.rand(self.batch_size,2))
        agentPosition1 = agentBound * (1 - 2 * torch.rand(self.batch_size,2))
        bladeBound = torch.zeros(self.batch_size, 2) + self.boundarySize - self.start_blade1.radius
        bladeMax1 = torch.min(agentPosition1 + 100, +bladeBound)
        bladeMin1 = torch.max(agentPosition1 - 100, -bladeBound)
        bladeRange1 = bladeMax1 - bladeMin1
        bladePosition1 = bladeMin1 + bladeRange1 * torch.rand(self.batch_size,2)
        self.start_agent0.position = torch.einsum('ij,kj->ki', self.rotation, agentPosition0)
        self.start_agent1.position = torch.einsum('ij,kj->ki', self.rotation, agentPosition1)
        self.start_blade1.position = torch.einsum('ij,kj->ki', self.rotation, bladePosition1)
        self.start_agent0.velocity = get_random_vectors(self.batch_size,30)
        self.start_agent1.velocity = get_random_vectors(self.batch_size,30)
        self.start_blade1.velocity = get_random_vectors(self.batch_size,70)
        self.state = get_simulation_state(self.start_simulation)

    def generate_outcomes(self):
        self.outcome_agent0.position = self.start_agent0.position.repeat_interleave(81, 0)
        self.outcome_agent0.velocity = self.start_agent0.velocity.repeat_interleave(81, 0)
        self.outcome_agent1.position = self.start_agent1.position.repeat_interleave(81, 0)
        self.outcome_agent1.velocity = self.start_agent1.velocity.repeat_interleave(81, 0)
        self.outcome_blade1.position = self.start_blade1.position.repeat_interleave(81, 0)
        self.outcome_agent1.velocity = self.start_blade1.velocity.repeat_interleave(81, 0)
        for _ in range(10):
            self.outcome_simulation.step()
        self.state = get_simulation_state(self.start_simulation)
        self.outcomes = get_simulation_state(self.outcome_simulation)
    
    # def generate(self)->tuple[Tensor,Tensor]:
    #     self.setup()
    #     startTensors = [
    #         self.agentPosition1 - self.agentPosition0,
    #         self.agentVelocity1,
    #         self.bladePosition1 - self.agentPosition0,
    #         self.bladeVelocity1,
    #         self.agentVelocity0,
    #         self.vision0,
    #     ]
    #     self.state = torch.cat(startTensors,dim=1)
    #     self.outcome_simulation.step()
    #     outcomeVision0 = visionCast(self.agent0.position, self.visionReach, self.outcome_simulation.boundary.walls)
    #     outcomeTensors = [
    #         self.agent1.position - self.agent0.position,
    #         self.agent1.velocity,
    #         self.blade1.position - self.agent1.position,
    #         self.blade1.velocity,
    #         self.agent0.velocity,
    #         outcomeVision0,
    #     ]
    #     self.outcomes = torch.cat(outcomeTensors,dim=1)
    #     return self.state, self.outcomes

        # def setup(self):
    #     agentBound = self.boundarySize - self.agent0.radius
    #     self.agentPosition0 = agentBound * (1 - 2 * torch.rand(self.count,2))
    #     self.agentPosition1 = agentBound * (1 - 2 * torch.rand(self.count,2))
    #     bladeBound = torch.zeros(self.count, 2) + self.boundarySize - self.blade1.radius
    #     bladeMax1 = torch.min(self.agentPosition1 + 100, +bladeBound)
    #     bladeMin1 = torch.max(self.agentPosition1 - 100, -bladeBound)
    #     bladeRange1 = bladeMax1 - bladeMin1
    #     self.bladePosition1 = bladeMin1 + bladeRange1 * torch.rand(self.count,2)
    #     self.agentVelocity0 = get_random_vectors(self.count,30)
    #     self.agentVelocity1 = get_random_vectors(self.count,30)
    #     self.bladeVelocity1 = get_random_vectors(self.count,70)
    #     angle = np.random.rand()*2*pi
    #     rotation = torch.tensor([
    #         [+cos(angle), -sin(angle)],
    #         [+sin(angle), +cos(angle)]
    #     ])
    #     boundaryPoints = torch.tensor([
    #         [-self.boundarySize,-self.boundarySize],
    #         [+self.boundarySize,-self.boundarySize],
    #         [+self.boundarySize,+self.boundarySize],
    #         [-self.boundarySize,+self.boundarySize]
    #     ],dtype=floatType)
    #     boundaryPoints = torch.einsum('ij,kj->ki', rotation, boundaryPoints)
    #     self.outcome_simulation.boundary.setup(boundaryPoints)
    #     self.agentPosition0 = torch.einsum('ij,kj->ki', rotation, self.agentPosition0)
    #     self.agentPosition1 = torch.einsum('ij,kj->ki', rotation, self.agentPosition1)
    #     self.bladePosition1 = torch.einsum('ij,kj->ki', rotation, self.bladePosition1)
    #     self.vision0 = visionCast(self.agentPosition0,self.visionReach,self.outcome_simulation.boundary.walls)
    #     self.agent0.position = self.agentPosition0.repeat_interleave(81, 0)
    #     self.agent0.velocity = self.agentVelocity0.repeat_interleave(81, 0)
    #     self.agent1.position = self.agentPosition1.repeat_interleave(81, 0)
    #     self.agent1.velocity = self.agentVelocity1.repeat_interleave(81, 0)
    #     self.blade1.position = self.bladePosition1.repeat_interleave(81, 0)
    #     self.blade1.velocity = self.bladeVelocity1.repeat_interleave(81, 0)

def get_random_directions(count: int)->Tensor:
    normals = torch.randn((count, 2))
    unit = F.normalize(normals,p=2,dim=1)
    return unit

def get_random_vectors(count: int, max_scale=1) ->Tensor:
    directions = get_random_directions(count)
    scales = max_scale*torch.rand(count).unsqueeze(1)
    return scales*directions

# TEST
# generator = DataGenerator()
# state, outcomes = generator.generate()
# print(state.shape)
# print(outcomes.shape)