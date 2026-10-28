from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import Tensor, tensor
from math import cos, pi, sin

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device = " + str(device))
physics_dtype = torch.float32
torch.set_default_device(device)
torch.set_printoptions(sci_mode=False, precision=4)

time_step = 0.02
max_speed = 200
move_power = 50
spring_power = 2
agent_radius = 15
blade_radius = 25
vision_range = 200

class Entity:
    def __init__(self, world: World):
        self.world = world
        self.index = len(world.entities)
        world.entities.append(self)

class Circle(Entity):
    def __init__(self, world: World, radius: int):
        super().__init__(world)
        self.world.circles.append(self)
        self.radius = radius
        self.mass = 1
        self.drag = 0
        self.position = torch.zeros(world.count,2,dtype=physics_dtype)
        self.velocity = torch.zeros(world.count,2,dtype=physics_dtype)
        self.force = torch.zeros(world.count,2,dtype=physics_dtype)
        self.impulse = torch.zeros(world.count,2,dtype=physics_dtype)
        self.shift = torch.zeros(world.count,2,dtype=physics_dtype)

class Agent(Circle):
    def __init__(self, world: World, align: int):
        super().__init__(world, agent_radius)
        self.alive = torch.ones(world.count, 1).bool()
        self.world.agents.append(self)
        self.align = align
        self.drag = 0.4
        self.action = torch.zeros(world.count,1).int()
        self.blade = Blade(self.world, self)

class Blade(Circle):
    def __init__(self, world: World, agent: Agent):
        super().__init__(world, blade_radius)
        self.world.blades.append(self)
        self.position = agent.position.detach().clone()
        self.agent = agent
        self.drag = 0.1

class Boundary:
    def __init__(self, world: World):
        self.world = world
        self.wall_starts: Tensor  # (world.count, num_walls, 2)
        self.wall_ends: Tensor    # (world.count, num_walls, 2)
        self.num_walls: int = 0
        world.boundaries.append(self)
    def setup(self, corners: Tensor):
        self.num_walls = corners.shape[-2]
        self.wall_starts = corners                        # (world.count, 4, 2)
        self.wall_ends = torch.roll(corners, -1, dims=1)  # (world.count, 4, 2)

action_vector_list = [[0.0,0.0]]
for i in range(8):
    angle = 2 * pi * i / 8
    vision_dir = [cos(angle), sin(angle)]
    action_vector_list.append(vision_dir)
action_tensor = torch.tensor(action_vector_list,dtype=physics_dtype)
actions = torch.tensor([i for i in range(9)])
action_count = actions.shape[0]

vision_dir_list: list[list[float]] = []
for i in range(8):
    angle = 2 * pi * i / 8
    vision_dir = [cos(angle), sin(angle)]
    vision_dir_list.append(vision_dir)
vision_dirs = torch.stack([torch.tensor(vd) for vd in vision_dir_list]).to(physics_dtype)

class World:
    def __init__(self, count: int):
        self.count = count
        self.device = device
        self.time = torch.zeros(self.count,1,dtype=physics_dtype)
        self.entities: list[Entity] = []
        self.circles: list[Circle] = []
        self.agents: list[Agent] = []
        self.blades: list[Blade] = []
        self.boundaries: list[Boundary] = []

    def step(self,dt: Tensor):
        for agent in self.agents:
            agent.force.fill_(0.0)
            agent.impulse.fill_(0.0)
            agent.shift.fill_(0.0)
        for blade in self.blades:
            blade.force.fill_(0.0)
            blade.impulse.fill_(0.0)
            blade.shift.fill_(0.0)
        for agent in self.agents:
            action = agent.action[:,0]
            agent.force = move_power * action_tensor[action,:]
        for blade in self.blades:
            blade.force = spring_power*(blade.agent.position-blade.position)
        for blade in self.blades:
            for otherBlade in self.blades:
                if blade.index < otherBlade.index:
                    collide_circle_circle(blade, otherBlade)
        for agent in self.agents:
            for otherAgent in self.agents:
                if agent.index < otherAgent.index:
                    collide_circle_circle(agent, otherAgent)
        for circle in self.circles:
            for boundary in self.boundaries:
                if boundary.num_walls > 0:
                    collide_circle_boundary(circle, boundary)
        self.time += dt
        ongoing = torch.where(dt>0,1,0)
        for circle in self.circles:
            circle.velocity = (1 - circle.drag * dt) * circle.velocity
            circle.velocity = circle.velocity + dt / circle.mass * circle.force
            circle.velocity = circle.velocity + ongoing*circle.impulse / circle.mass
            speed = torch.norm(circle.velocity, dim=1, keepdim=True)
            max_velocity = max_speed*F.normalize(circle.velocity, dim=1)
            circle.velocity = torch.where(speed>max_speed,max_velocity,circle.velocity)
            circle.position = circle.position + dt * circle.velocity + ongoing*circle.shift

def collide_circle_circle(circle1: Circle, circle2: Circle):
    if circle1.index >= circle2.index: return
    vector = circle2.position - circle1.position
    distance = torch.sqrt(torch.sum(vector ** 2, dim=1))
    overlap = (circle1.radius + circle2.radius - distance).unsqueeze(1)
    normal = F.normalize(vector, dim=1)
    relative_velocity = circle1.velocity - circle2.velocity
    impact_speed = torch.linalg.vecdot(relative_velocity, normal).unsqueeze(1)
    impact_speed = torch.where(impact_speed > 0, impact_speed, 0)
    mass_factor = 1 / circle1.mass + 1 / circle2.mass
    impulse = torch.where(overlap > 0, impact_speed / mass_factor * normal, 0)
    shift = torch.where(overlap > 0, 0.5 * overlap * normal, 0)
    circle1.impulse = circle1.impulse - impulse
    circle2.impulse = circle2.impulse + impulse
    circle1.shift = circle1.shift - shift
    circle2.shift = circle2.shift + shift

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
    impact_speed = torch.where(impact_speed > 0, impact_speed, 0)
    impulse = torch.where(hit_mask, impact_speed * circle.mass * normal, 0.0)
    shift = torch.where(hit_mask, overlap * normal, 0.0)
    circle.impulse += torch.sum(impulse, dim=1)
    circle.shift += torch.sum(shift, dim=1)

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

def vision_cast(circle: Circle)->Tensor:
    min_walls = min([b.num_walls for b in circle.world.boundaries],default=0)
    if min_walls == 0: return torch.ones(circle.world.count,8)
    wall_starts = torch.cat([b.wall_starts for b in circle.world.boundaries],dim=1)
    wall_ends = torch.cat([b.wall_ends for b in circle.world.boundaries],dim=1)
    ray_factors = raycast_segment(
        circle.position[:, None, None, :],   # (N, 1, 1, 2)
        (vision_range * vision_dirs)[None, :, None, :],   # (1, 8, 1, 2)
        wall_starts[:, None, :, :],          # (N, 1, W, 2)
        wall_ends[:, None, :, :],
    )               
    min_ray_factors = torch.amin(ray_factors, dim=-1)
    clamp_ray_factors = torch.clamp(min_ray_factors, min=0.0, max=1.0)
    return clamp_ray_factors