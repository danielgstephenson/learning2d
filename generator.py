import torch

from physics import Agent, Blade, Boundary, Simulation, actions


class Generator:
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
        self.agent0 = Agent(self.simulation, 0)
        self.agent1 = Agent(self.simulation, 1)
        self.blade0 = Blade(self.simulation, self.agent0)
        self.blade1 = Blade(self.simulation, self.agent1)
        self.action0 = actions.repeat_interleave(9, dim=0)
        self.action1 = actions.repeat(9)
        self.agent0.action = self.action0.repeat(self.count)
        self.agent1.action = self.action1.repeat(self.count)

    def setup(self):
        agentBound = self.size - self.agent0.radius
        agentPosition0 = agentBound * (1 - 2 * torch.rand(self.count,2))
        agentPosition1 = agentBound * (1 - 2 * torch.rand(self.count,2))
        bladeBound = torch.zeros_like(agentPosition0) + self.size - self.blade0.radius
        bladeMax0 = torch.min(agentPosition0 + 100, +bladeBound)
        bladeMin0 = torch.max(agentPosition0 - 100, -bladeBound)
        bladeRange0 = bladeMax0 - bladeMin0
        bladePosition0 = bladeMin0 + bladeRange0 * torch.rand(self.count,2)
        bladeMax1 = torch.min(agentPosition1 + 100, +bladeBound)
        bladeMin1 = torch.max(agentPosition1 - 100, -bladeBound)
        bladeRange1 = bladeMax1 - bladeMin1
        bladePosition1 = bladeMin1 + bladeRange1 * torch.rand(self.count,2)
        self.agent0.position = agentPosition0.repeat_interleave(81, 0)
        self.agent1.position = agentPosition1.repeat_interleave(81, 0)
        self.blade0.position = bladePosition0.repeat_interleave(81, 0)
        self.blade1.position = bladePosition1.repeat_interleave(81, 0)
        # NEXT: Ranomdize initial velocities
        self.agent0.velocity = 0*self.agent0.velocity
        self.agent1.velocity = 0*self.agent1.velocity
        self.blade0.velocity = 0*self.blade0.velocity
        self.blade1.velocity = 0*self.blade1.velocity