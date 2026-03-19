import torch
import torch.nn.functional as F

from simulation import Circle

def collideCircleCircle(circle1: Circle, circle2: Circle):
    if circle1.index >= circle2.index: return
    vector = circle2.position - circle1.position
    distance = torch.linalg.norm(vector, dim=1)
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
