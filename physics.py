from __future__ import annotations
from numpy import dtype, number, where
import torch
import torch.nn.functional as F
from torch import Tensor
from math import cos, inf, pi, sin

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device = " + str(device))
physics_dtype = torch.float32
torch.set_default_device(device)

class Entity:
    def __init__(self, simulation: Simulation):
        self.simulation = simulation
        self.index = len(simulation.entities)
        simulation.entities.append(self)

class Circle(Entity):
    def __init__(self, simulation: Simulation, radius: int):
        super().__init__(simulation)
        self.simulation.circles.append(self)
        self.radius = radius
        self.mass = 0.01 * pi * radius ** 2
        self.drag = 0
        self.position = torch.zeros(simulation.count,2,dtype=physics_dtype)
        self.velocity = torch.zeros(simulation.count,2,dtype=physics_dtype)
        self.force = torch.zeros(simulation.count,2,dtype=physics_dtype)
        self.impulse = torch.zeros(simulation.count,2,dtype=physics_dtype)
        self.shift = torch.zeros(simulation.count,2,dtype=physics_dtype)

class Agent(Circle):
    def __init__(self, simulation: Simulation, align: int):
        super().__init__(simulation, 5)
        self.simulation.agents.append(self)
        self.align = align
        self.mass = 1
        self.drag = 0.7
        self.move_power = 20
        self.action = torch.zeros(simulation.count, dtype=torch.int)

class Blade(Circle):
    def __init__(self, simulation: Simulation, agent: Agent):
        super().__init__(simulation, 10)
        self.simulation.blades.append(self)
        self.position = agent.position.detach().clone()
        self.agent = agent
        self.move_power = 2
        self.drag = 0.3

class Boundary():
    def __init__(self, simulation: Simulation):
        self.simulation = simulation
        simulation.boundary = self
        self.walls: list[list[Tensor]] = []
        self.corners: list[Tensor] = []

    def setup(self, points: list[Tensor]):
        self.walls = []
        self.corners = []
        n = len(points)
        points = [x.to(physics_dtype) for x in points]
        for i in range(n):
            j = i - 1 if i > 0 else n - 1
            self.corners.append(points[i])
            self.walls.append([points[i],points[j]])

actionVectorList = [[0.0,0.0]]
for i in range(8):
    angle = 2 * pi * i / 8
    visionDir = [cos(angle), sin(angle)]
    actionVectorList.append(visionDir)
actionVectors = torch.tensor(actionVectorList,dtype=physics_dtype).to(device)
actions = torch.tensor([i for i in range(9)]).to(device)

visionDirList: list[list[float]] = []
for i in range(8):
    angle = 2 * pi * i / 8
    visionDir = [cos(angle), sin(angle)]
    visionDirList.append(visionDir)
visionDirs: list[Tensor] = [
    torch.tensor(visionDir,dtype=physics_dtype).to(device)
    for visionDir in visionDirList
]

class Simulation:
    def __init__(self, count: int, timeStep):
        self.count = count
        self.device = device
        self.timeStep = timeStep
        self.dtype = dtype
        self.entities: list[Entity] = []
        self.circles: list[Circle] = []
        self.agents: list[Agent] = []
        self.blades: list[Blade] = []
        self.boundary: Boundary = Boundary(self)

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
            agent.force = agent.move_power * actionVectors[agent.action]
        for blade in self.blades:
            vector = blade.agent.position - blade.position
            blade.force = blade.move_power * vector
        for blade1 in self.blades:
            for blade2 in self.blades:
                collideCircleCircle(blade1, blade2)
        for agent1 in self.agents:
            for agent2 in self.agents:
                collideCircleCircle(agent1, agent2)
        for circle in self.circles:
            for corner in self.boundary.corners:
                collideCirclePoint(circle, corner)
            for wall in self.boundary.walls:
                collideCircleSegment(circle, wall)
        dt = self.timeStep
        for circle in self.circles:
            circle.velocity = (1 - circle.drag * dt) * circle.velocity 
            circle.velocity = circle.velocity + dt / circle.mass * circle.force
            circle.velocity = circle.velocity + 1 / circle.mass * circle.impulse
            circle.position = circle.position + dt * circle.velocity + circle.shift

def collideCircleCircle(circle1: Circle, circle2: Circle):
    if circle1.index >= circle2.index: return
    vector = circle2.position - circle1.position
    distance = torch.sqrt(torch.sum(vector ** 2, dim=1))
    overlap = (circle1.radius + circle2.radius - distance).unsqueeze(1)
    normal = F.normalize(vector, dim=1)
    relativeVelocity = circle1.velocity - circle2.velocity
    impactSpeed = torch.linalg.vecdot(relativeVelocity, normal).unsqueeze(1)
    massFactor = 1 / circle1.mass + 1 / circle2.mass
    impulse = torch.where(overlap > 0, impactSpeed / massFactor * normal, 0)
    shift = torch.where(overlap > 0, 0.5 * overlap * normal, 0)
    circle1.impulse = circle1.impulse - impulse
    circle2.impulse = circle2.impulse + impulse
    circle1.shift = circle1.shift - shift
    circle2.shift = circle2.shift + shift

def collideCirclePoint(circle: Circle, point: Tensor):
    vector = torch.sub(circle.position, point)
    distance = torch.sqrt(torch.sum(vector ** 2, dim=1)).unsqueeze(1)
    overlap = (circle.radius - distance)
    normal = F.normalize(vector)
    impactSpeed = -torch.einsum('ij,ij->i',circle.velocity, normal).unsqueeze(1)
    circle.impulse += torch.where(overlap > 0, 1.2 * impactSpeed * circle.mass * normal, 0)
    circle.shift += torch.where(overlap > 0, overlap * normal, 0)

def collideCircleSegment(circle: Circle, segment: list[Tensor]):
    a = segment[0]
    b = segment[1]
    c = circle.position
    ab = b-a
    ac = c-a
    bc = c-b
    sideDot0 = torch.einsum('ij,ij->i',ac,+ab).unsqueeze(1)
    sideDot1 = torch.einsum('ij,ij->i',bc,-ab).unsqueeze(1)
    segmentDir = F.normalize(ab,dim=1)
    normal0 = torch.stack((-segmentDir[:,1],+segmentDir[:,0]),dim=1)
    normal1 = -normal0
    normalDot0 = torch.einsum('ij,ij->i',ac,normal0).unsqueeze(1)
    normal = torch.where(normalDot0 > 0, normal0, normal1)
    normalDot = torch.abs(normalDot0)
    hit = (sideDot0 > 0) & (sideDot1 > 0) & (circle.radius > normalDot)
    overlap = torch.where(hit, circle.radius - normalDot, 0)
    impactSpeed = torch.einsum('ij,ij->i',circle.velocity,-normal).unsqueeze(1)
    impulse = 1.2 * impactSpeed * circle.mass * normal
    circle.impulse += torch.where(overlap > 0, impulse, 0)
    shift = overlap * normal
    circle.shift += shift

def cross2d(v0: Tensor, v1: Tensor):
    x0, y0 = v0[..., 0], v0[..., 1]
    x1, y1 = v1[..., 0], v1[..., 1]
    return x0 * y1 - y0 * x1

def rayCastSegment(rayStart: Tensor, rayVector: Tensor, segment: list[Tensor])->Tensor:
    segmentStart = segment[0]
    segmentEnd = segment[1]
    segmentVector = segmentEnd - segmentStart
    startDifference = segmentStart - rayStart
    denominator = cross2d(rayVector, segmentVector)
    rayFactor = torch.where(denominator != 0, cross2d(startDifference, segmentVector) / denominator, 0)
    rayHit = rayFactor > 0
    segmentFactor = torch.where(denominator != 0, cross2d(startDifference, rayVector) / denominator, 0)
    segmentHit = (0 < segmentFactor) & (segmentFactor < 1)
    return torch.where(rayHit & segmentHit, rayFactor, inf)

def rayCastSegments(rayStart: Tensor, rayVector: Tensor, segments: list[list[Tensor]])->Tensor:
    rayCount = rayStart.shape[0]
    segmentCount = len(segments)
    rayFactorMatrix = torch.zeros(rayCount, segmentCount) + inf
    for j in range(segmentCount):
        segment = segments[j]
        rayFactorMatrix[:,j] = rayCastSegment(rayStart, rayVector, segment)
    rayFactors = torch.amin(rayFactorMatrix,dim=1)
    return rayFactors
    
def visionCast(origin: Tensor, reach: float, walls: list[list[Tensor]])->Tensor:
    rayFactorColumns: list[Tensor] = []
    for lookDir in visionDirs:
        lookVector = reach*lookDir
        rayFactors = rayCastSegments(origin, lookVector, walls)
        rayFactors = torch.where(rayFactors < 1, rayFactors, 1)
        rayFactorColumns.append(reach * rayFactors)
    rayFectorMatrix = torch.stack(rayFactorColumns,1)
    return rayFectorMatrix

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