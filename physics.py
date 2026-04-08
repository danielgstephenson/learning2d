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
torch.set_printoptions(sci_mode=False, precision=4)

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

action_vector_list = [[0.0,0.0]]
for i in range(8):
    angle = 2 * pi * i / 8
    vision_dir = [cos(angle), sin(angle)]
    action_vector_list.append(vision_dir)
action_tensor = torch.tensor(action_vector_list,dtype=physics_dtype).to(device)
actions = torch.tensor([i for i in range(9)]).to(device)

vision_dir_list: list[list[float]] = []
for i in range(8):
    angle = 2 * pi * i / 8
    vision_dir = [cos(angle), sin(angle)]
    vision_dir_list.append(vision_dir)
vision_dirs: list[Tensor] = [
    torch.tensor(visionDir,dtype=physics_dtype).to(device)
    for visionDir in vision_dir_list
]

class Simulation:
    def __init__(self, count: int, timeStep):
        self.count = count
        self.device = device
        self.time_step = timeStep
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
            agent.force = agent.move_power * action_tensor[agent.action]
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
        dt = self.time_step
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
    relative_velocity = circle1.velocity - circle2.velocity
    impact_speed = torch.linalg.vecdot(relative_velocity, normal).unsqueeze(1)
    mass_factor = 1 / circle1.mass + 1 / circle2.mass
    impulse = torch.where(overlap > 0, impact_speed / mass_factor * normal, 0)
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
    impact_speed = -torch.einsum('ij,ij->i',circle.velocity, normal).unsqueeze(1)
    circle.impulse += torch.where(overlap > 0, 1.2 * impact_speed * circle.mass * normal, 0)
    circle.shift += torch.where(overlap > 0, overlap * normal, 0)

def collideCircleSegment(circle: Circle, segment: list[Tensor]):
    a = segment[0]
    b = segment[1]
    c = circle.position
    ab = b-a
    ac = c-a
    bc = c-b
    side_dot0 = torch.einsum('ij,ij->i',ac,+ab).unsqueeze(1)
    side_dot1 = torch.einsum('ij,ij->i',bc,-ab).unsqueeze(1)
    segment_dir = F.normalize(ab,dim=1)
    normal0 = torch.stack((-segment_dir[:,1],+segment_dir[:,0]),dim=1)
    normal1 = -normal0
    normal_dot0 = torch.einsum('ij,ij->i',ac,normal0).unsqueeze(1)
    normal = torch.where(normal_dot0 > 0, normal0, normal1)
    normal_dot = torch.abs(normal_dot0)
    hit = (side_dot0 > 0) & (side_dot1 > 0) & (circle.radius > normal_dot)
    overlap = torch.where(hit, circle.radius - normal_dot, 0)
    impact_speed = torch.einsum('ij,ij->i',circle.velocity,-normal).unsqueeze(1)
    impulse = 1.2 * impact_speed * circle.mass * normal
    circle.impulse += torch.where(overlap > 0, impulse, 0)
    shift = overlap * normal
    circle.shift += shift

def cross2d(v0: Tensor, v1: Tensor):
    x0, y0 = v0[..., 0], v0[..., 1]
    x1, y1 = v1[..., 0], v1[..., 1]
    return x0 * y1 - y0 * x1

def rayCastSegment(rayStart: Tensor, rayVector: Tensor, segment: list[Tensor])->Tensor:
    segment_start = segment[0]
    segment_end = segment[1]
    segment_vector = segment_end - segment_start
    start_difference = segment_start - rayStart
    denominator = cross2d(rayVector, segment_vector)
    ray_factor = torch.where(denominator != 0, cross2d(start_difference, segment_vector) / denominator, 0)
    ray_hit = ray_factor > 0
    segmentFactor = torch.where(denominator != 0, cross2d(start_difference, rayVector) / denominator, 0)
    segmentHit = (0 < segmentFactor) & (segmentFactor < 1)
    return torch.where(ray_hit & segmentHit, ray_factor, inf)

def rayCastSegments(rayStart: Tensor, rayVector: Tensor, segments: list[list[Tensor]])->Tensor:
    ray_count = rayStart.shape[0]
    segment_count = len(segments)
    ray_factor_matrix = torch.zeros(ray_count, segment_count) + inf
    for j in range(segment_count):
        segment = segments[j]
        ray_factor_matrix[:,j] = rayCastSegment(rayStart, rayVector, segment)
    ray_factors = torch.amin(ray_factor_matrix,dim=1)
    return ray_factors
    
def visionCast(origin: Tensor, reach: float, walls: list[list[Tensor]])->Tensor:
    ray_factor_columns: list[Tensor] = []
    for look_dir in vision_dirs:
        look_vector = reach*look_dir
        ray_factors = rayCastSegments(origin, look_vector, walls)
        ray_factors = torch.where(ray_factors < 1, ray_factors, 1)
        ray_factor_columns.append(reach * ray_factors)
    ray_factor_matrix = torch.stack(ray_factor_columns,1)
    return ray_factor_matrix

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