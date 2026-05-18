from __future__ import annotations
from numpy import dtype
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
        self.wall_starts = torch.empty(0)
        self.wall_ends = torch.empty(0)
        self.num_walls = 0

    def setup(self, points: Tensor):
        self.wall_starts = points.to(physics_dtype)
        self.wall_ends = torch.roll(self.wall_starts, shifts=-1, dims=1)
        self.num_walls = self.wall_starts.shape[1]

action_vector_list = [[0.0,0.0]]
for i in range(8):
    angle = 2 * pi * i / 8
    vision_dir = [cos(angle), sin(angle)]
    action_vector_list.append(vision_dir)
action_tensor = torch.tensor(action_vector_list,dtype=physics_dtype)
actions = torch.tensor([i for i in range(9)])
active_action_tensor = action_tensor[1:,:]

vision_dir_list: list[list[float]] = []
for i in range(8):
    angle = 2 * pi * i / 8
    vision_dir = [cos(angle), sin(angle)]
    vision_dir_list.append(vision_dir)
vision_dirs = torch.stack([torch.tensor(vd) for vd in vision_dir_list])

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
            agent.force = agent.move_power * action_tensor[agent.action,:]
        for blade in self.blades:
            vector = blade.agent.position - blade.position
            blade.force = blade.move_power * vector
        for blade1 in self.blades:
            for blade2 in self.blades:
                collide_circle_circle(blade1, blade2)
        for agent1 in self.agents:
            for agent2 in self.agents:
                collide_circle_circle(agent1, agent2)
        for circle in self.circles:
            collide_circle_boundary(circle, self.boundary)
        dt = self.time_step
        for circle in self.circles:
            circle.velocity = (1 - circle.drag * dt) * circle.velocity 
            circle.velocity = circle.velocity + dt / circle.mass * circle.force
            circle.velocity = circle.velocity + 1 / circle.mass * circle.impulse
            circle.position = circle.position + dt * circle.velocity + circle.shift

def collide_circle_circle(circle1: Circle, circle2: Circle):
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

def collide_circle_point(circle: Circle, point: Tensor):
    vector = torch.sub(circle.position, point)
    distance = torch.sqrt(torch.sum(vector ** 2, dim=1)).unsqueeze(1)
    overlap = (circle.radius - distance)
    normal = F.normalize(vector)
    impact_speed = -torch.einsum('ij,ij->i',circle.velocity, normal).unsqueeze(1)
    circle.impulse += torch.where(overlap > 0, 1.2 * impact_speed * circle.mass * normal, 0)
    circle.shift += torch.where(overlap > 0, overlap * normal, 0)

def collide_circle_boundary(circle: Circle, boundary: Boundary):
    position = circle.position.unsqueeze(1)
    segment_vector = boundary.wall_ends - boundary.wall_starts
    relative_position = position - boundary.wall_starts
    squared_segment_length = torch.sum(segment_vector**2, dim=-1, keepdim=True)
    segment_factor = torch.sum(relative_position * segment_vector, dim=-1, keepdim=True) / (squared_segment_length + 1e-9)
    segment_factor = torch.clamp(segment_factor, 0.0, 1.0)
    closest_point = boundary.wall_starts + segment_factor * segment_vector
    vector_to_circle = position - closest_point
    squared_distance = torch.sum(vector_to_circle**2, dim=-1, keepdim=True)
    distance = torch.sqrt(squared_distance + 1e-9)
    overlap = circle.radius - distance
    hit_mask = overlap > 0
    normal = vector_to_circle / distance
    velocity = circle.velocity.unsqueeze(1)
    impact_speed = -torch.sum(velocity * normal, dim=-1, keepdim=True)
    impulse = torch.where(hit_mask, 1.2 * impact_speed * circle.mass * normal, 0.0)
    shift = torch.where(hit_mask, overlap * normal, 0.0)
    circle.impulse += torch.sum(impulse, dim=1)
    circle.shift += torch.sum(shift, dim=1)

def collide_circle_segment(circle: Circle, segment: list[Tensor]):
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

def cross2d(v0: Tensor, v1: Tensor)->Tensor:
    return v0[..., 0] * v1[..., 1] - v0[..., 1] * v1[..., 0]

def raycast_segment(ray_start: Tensor, ray_vector: Tensor, segment_start: Tensor, segment_end: Tensor)->Tensor:
    segment_vector = segment_end - segment_start
    start_difference = segment_start - ray_start
    denominator = cross2d(ray_vector, segment_vector)
    ray_factor = cross2d(start_difference, segment_vector) / (denominator + 1e-9)
    segment_factor = torch.where(denominator != 0, cross2d(start_difference, ray_vector) / denominator, 0)
    hit = (denominator != 0) & (ray_factor >= 0) & (segment_factor >= 0) & (segment_factor <= 1)
    inf_tensor = torch.full_like(ray_factor, float('inf'))
    return torch.where(hit, ray_factor, inf_tensor)

def vision_cast(origin: Tensor, reach: float | int, boundary: Boundary)->Tensor:
    batch_size = origin.shape[0]
    ray_vectors = (reach*vision_dirs).expand(batch_size, -1, -1)
    vmap_raycast = torch.vmap(
        torch.vmap(         
            torch.vmap(
                raycast_segment, 
                in_dims=(None, None, 0, 0)
            ),
            in_dims=(None, 0, None, None) 
        ),
        in_dims=(0, 0, 0, 0)
    )
    ray_factors = vmap_raycast(origin, ray_vectors, boundary.wall_starts, boundary.wall_ends)
    min_ray_factors = torch.amin(ray_factors, dim=-1)
    clamp_ray_factors = torch.clamp(min_ray_factors, min=0.0, max=1.0).unsqueeze(-1)
    relative_hitpoints = clamp_ray_factors * ray_vectors
    return relative_hitpoints