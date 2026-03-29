from typing import Sequence
from sympy import sequence
import torch
from torch import Tensor
import torch.nn.functional as F

from physics import Agent, Blade, Boundary, Simulation, actions, visionCast

class DataGenerator:
    def __init__(self, count = 3, timeStep = 0.1):
        self.count = count
        self.simulation = Simulation(81 * count, timeStep)
        self.size = 200
        self.boundary = Boundary(self.simulation,[
            [-self.size,-self.size],
            [+self.size,-self.size],
            [+self.size,+self.size],
            [-self.size,+self.size]
        ])
        self.visionReach = 100
        self.agent0 = Agent(self.simulation, 0)
        self.agent1 = Agent(self.simulation, 1)
        self.blade1 = Blade(self.simulation, self.agent1)
        self.action0 = actions.repeat_interleave(9, dim=0)
        self.action1 = actions.repeat(9)
        self.agent0.action = self.action0.repeat(self.count)
        self.agent1.action = self.action1.repeat(self.count)
        self.agentPosition0: Tensor
        self.agentVelocity0: Tensor
        self.agentPosition1: Tensor
        self.agentVelocity1: Tensor
        self.bladePosition1: Tensor
        self.bladeVelocity1: Tensor
        self.vision0: Tensor

    def setup(self):
        agentBound = self.size - self.agent0.radius
        self.agentPosition0 = agentBound * (1 - 2 * torch.rand(self.count,2))
        self.agentPosition1 = agentBound * (1 - 2 * torch.rand(self.count,2))
        self.vision0 = visionCast(self.agentPosition0,self.visionReach,self.simulation)
        bladeBound = torch.zeros(self.count, 2) + self.size - self.blade1.radius
        bladeMax1 = torch.min(self.agentPosition1 + 100, +bladeBound)
        bladeMin1 = torch.max(self.agentPosition1 - 100, -bladeBound)
        bladeRange1 = bladeMax1 - bladeMin1
        self.bladePosition1 = bladeMin1 + bladeRange1 * torch.rand(self.count,2)
        self.agentVelocity0 = get_random_vectors(self.count,30)
        self.agentVelocity1 = get_random_vectors(self.count,30)
        self.bladeVelocity1 = get_random_vectors(self.count,70)
        self.agent0.position = self.agentPosition0.repeat_interleave(81, 0)
        self.agent0.velocity = self.agentVelocity0.repeat_interleave(81, 0)
        self.agent1.position = self.agentPosition1.repeat_interleave(81, 0)
        self.agent1.velocity = self.agentVelocity1.repeat_interleave(81, 0)
        self.blade1.position = self.bladePosition1.repeat_interleave(81, 0)
        self.blade1.velocity = self.bladeVelocity1.repeat_interleave(81, 0)
    
    def generate(self)->tuple[Tensor,Tensor]:
        self.setup()
        startTensors = [
            self.agentPosition1 - self.agentPosition0,
            self.agentVelocity1,
            self.bladePosition1 - self.agentPosition0,
            self.bladeVelocity1,
            self.agentVelocity0,
            self.vision0,
        ]
        state = torch.cat(startTensors,dim=1)
        self.simulation.step()
        outcomeAgentPosistion1 = self.agent1.position
        outcomeVision1 = visionCast(outcomeAgentPosistion1, self.visionReach, self.simulation)
        outcomeTensors = [
            self.agent1.position - self.agent0.position,
            self.agent1.velocity,
            self.blade1.position - self.agent1.position,
            self.blade1.velocity,
            self.agent0.velocity,
            outcomeVision1,
        ]
        outcomes = torch.cat(outcomeTensors,dim=1)
        return state, outcomes

def get_random_directions(count: int)->Tensor:
    normals = torch.randn((count, 2))
    unit = F.normalize(normals,p=2,dim=0)
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