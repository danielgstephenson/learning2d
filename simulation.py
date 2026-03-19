from __future__ import annotations
import torch
from math import cos, pi, sin

from collide import collideCircleCircle

class Entity:
    def __init__(self, simulation: Simulation):
        self.simulation = simulation
        self.index = len(simulation.entities)
        simulation.entities.append(self)

class Circle(Entity):
    def __init__(self, simulation: Simulation, radius: float):
        super().__init__(simulation)
        self.simulation.circles.append(self)
        self.radius = radius
        self.mass = 0.01 * pi * radius ** 2
        self.drag = 0
        self.position = torch.zeros(simulation.count,2,dtype=simulation.dtype).to(simulation.device)
        self.velocity = torch.zeros(simulation.count,2,dtype=simulation.dtype).to(simulation.device)
        self.force = torch.zeros(simulation.count,2,dtype=simulation.dtype).to(simulation.device)
        self.impulse = torch.zeros(simulation.count,2,dtype=simulation.dtype).to(simulation.device)
        self.shift = torch.zeros(simulation.count,2,dtype=simulation.dtype).to(simulation.device)

class Agent(Circle):
    def __init__(self, simulation: Simulation, align: int):
        super().__init__(simulation, 5)
        self.simulation.agents.append(self)
        self.align = align
        self.mass = 1
        self.drag = 0.7
        self.move_power = 20
        self.dead = torch.zeros(simulation.count, dtype=torch.int).to(simulation.device)
        self.action = torch.zeros(simulation.count, dtype=torch.int).to(simulation.device)

class Blade(Circle):
    def __init__(self, simulation: Simulation, agent: Agent):
        super().__init__(simulation, 5)
        self.simulation.blades.append(self)
        self.agent = agent
        self.move_power = 2
        self.drag = 0.3

class Simulation:
    def __init__(self, count: int, device: torch.device, timeStep=0.04, dtype = torch.float32):
        self.count = count
        self.device = device
        self.timeStep = timeStep
        self.dtype = dtype
        self.entities: list[Entity] = []
        self.circles: list[Circle] = []
        self.agents: list[Agent] = []
        self.blades: list[Blade] = []
        action_vector_list = [[0.0,0.0]]
        for i in range(8):
            angle = 2 * pi * i / 8
            dir = [cos(angle), sin(angle)]
            action_vector_list.append(dir)
        self.action_tensor = torch.tensor(action_vector_list).to(device)
        self.actions = torch.tensor([i for i in range(9)]).to(device)

    def step(self):
        for agent in self.agents:
            agent.force[:,:] = 0
            agent.impulse[:,:] = 0
            agent.shift[:,:] = 0
        for blade in self.blades:
            blade.force[:,:] = 0
            blade.impulse[:,:] = 0
            blade.shift[:,:] = 0
        for agent in self.agents:
            agent.force = agent.move_power * self.action_tensor[agent.action]
        for blade in self.blades:
            vector = blade.agent.position - blade.position
            blade.force = blade.move_power * vector
        for blade1 in self.blades:
            for blade2 in self.blades:
                collideCircleCircle(blade1, blade2)
        for agent1 in self.agents:
            for agent2 in self.agents:
                collideCircleCircle(agent1, agent2)
        dt = self.timeStep
        for circle in self.circles:
            circle.velocity = (1 - circle.drag * dt) * circle.velocity 
            circle.velocity = circle.velocity + dt / circle.mass * circle.force
            circle.velocity = circle.velocity + 1 / circle.mass * circle.impulse
            circle.position = circle.position + dt * circle.velocity + circle.shift

        