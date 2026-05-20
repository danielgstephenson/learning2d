from math import pi
from numpy import where
import torch
from torch import Tensor
from torch.func import vmap, grad
import torch.nn.functional as F

from value import ValueModel
from physics import Agent, Blade, Simulation, active_action_tensor, physics_dtype

unit_square = torch.tensor([[-1,-1],[1,-1],[1,1],[-1,1]]).to(physics_dtype)

class DataGenerator:
    def __init__(self, value_model: ValueModel, batch_size = 3, time_step = 0.1, step_count = 50):
        self.value_model = value_model
        self.batch_size = batch_size
        self.step_count = step_count
        self.time_step = time_step
        self.simulation = Simulation(batch_size, time_step)
        self.get_costate = vmap(grad(lambda x: self.value_model(x).sum()))
        self.agent0 = Agent(self.simulation, 0)
        self.blade0 = Blade(self.simulation, self.agent0)
        self.agent1 = Agent(self.simulation, 1)
        self.blade1 = Blade(self.simulation, self.agent1)
        self.scale: Tensor
        self.state: Tensor
        self.costate: Tensor
        self.vgrad0: Tensor
        self.vgrad1: Tensor
        self.life0: Tensor
        self.life1: Tensor
        self.ringDistance: Tensor
        self.reset()
    
    def reset(self):
        staticDistance = (20 + 10*torch.rand(self.batch_size).unsqueeze(1))
        staticAgent0Position = staticDistance*get_random_directions(self.batch_size)
        staticBlade0Position = staticAgent0Position + get_random_vectors(self.batch_size, 0)
        staticAgent1Position = get_random_vectors(self.batch_size,0)
        staticBlade1Position = get_random_vectors(self.batch_size,0)
        staticAgent0Velocity = get_random_vectors(self.batch_size,0)
        staticBlade0Velocity = get_random_vectors(self.batch_size,0)
        staticAgent1Velocity = get_random_vectors(self.batch_size,0)
        staticBlade1Velocity = get_random_vectors(self.batch_size,0)
        dynamicAgent0Position = get_random_vectors(self.batch_size, 100)
        dynamicBlade0Position = dynamicAgent0Position + get_random_vectors(self.batch_size, 80)
        dynamicAgent1Position = get_random_vectors(self.batch_size, 100)
        dynamicBlade1Position = dynamicAgent1Position + get_random_vectors(self.batch_size, 80)
        dynamicAgent0Velocity = get_random_vectors(self.batch_size,30)
        dynamicBlade0Velocity = get_random_vectors(self.batch_size,70)
        dynamicAgent1Velocity = get_random_vectors(self.batch_size,30)
        dynamicBlade1Velocity = get_random_vectors(self.batch_size,70)
        static = torch.rand(self.batch_size) < 0.5
        static = torch.stack((static,static),dim=1)
        self.agent0.position = torch.where(static, staticAgent0Position, dynamicAgent0Position)
        self.blade0.position = torch.where(static, staticBlade0Position, dynamicBlade0Position)
        self.agent1.position = torch.where(static, staticAgent1Position, dynamicAgent1Position)
        self.blade1.position = torch.where(static, staticBlade1Position, dynamicBlade1Position)
        self.agent0.velocity = torch.where(static, staticAgent0Velocity, dynamicAgent0Velocity)
        self.blade0.velocity = torch.where(static, staticBlade0Velocity, dynamicBlade0Velocity)
        self.agent1.velocity = torch.where(static, staticAgent1Velocity, dynamicAgent1Velocity)
        self.blade1.velocity = torch.where(static, staticBlade1Velocity, dynamicBlade1Velocity)

    def update(self, horizon: int):
        self.state = get_simulation_state(self.simulation)
        gap0 = torch.norm(self.agent0.position - self.blade1.position,p=2,dim=1,keepdim=True)
        gap1 = torch.norm(self.agent1.position - self.blade0.position,p=2,dim=1,keepdim=True)
        self.life0 = torch.where(gap0 > 15, 1, 0).to(physics_dtype)
        self.life1 = torch.where(gap1 > 15, 1, 0).to(physics_dtype)
        self.ringDistance = torch.norm(self.agent1.position,p=2,dim=1,keepdim=True)
        if horizon==0:
            self.costate = 0*self.state
            self.vgrad0 = +self.costate[:,[0,1]]
            self.agent0.action = torch.zeros(self.batch_size).int()
            self.agent1.action = torch.zeros(self.batch_size).int()
        else:
            self.costate = self.get_costate(self.state)
            self.vgrad0 = +self.costate[:,[0,1]]
            self.vgrad1 = -self.costate[:,[8,9]]
            action_values0 = torch.einsum('ij,kj->ik',self.vgrad0,active_action_tensor)
            action_values1 = torch.einsum('ij,kj->ik',self.vgrad1,active_action_tensor)
            self.agent0.action = torch.argmax(action_values0, dim=1)+1
            self.agent1.action = torch.argmax(action_values1, dim=1)+1

    def getReward(self, life0: Tensor, life1: Tensor, ringDistance: Tensor)->Tensor:
        victory = life0 * (1 - life1) > 0
        ringOut = ringDistance > 20
        return torch.where(victory | ringOut, 1, 0)
        
    def generate(self, horizon: int)->tuple[Tensor,...]:
        self.value_model.eval()
        p = 0.005 # Discount Rate
        with torch.no_grad():
            self.reset()
            self.update(horizon)
            state = self.state.clone()
            life0 = self.life0.clone()
            life1 = self.life1.clone()
            reward = self.getReward(life0, life1, self.ringDistance)
            value_target = reward * p
            for t in range(self.step_count):
                self.simulation.step()
                self.update(horizon)
                complete = (life0*life1 == 0)
                life0 = torch.where(complete, life0, life0*self.life0)
                life1 = torch.where(complete, life1, life1*self.life1)
                end_prob = p * (1-p) ** (t+1)    
                reward = self.getReward(life0, life1, self.ringDistance)
                value_target += reward * end_prob
            outcome = self.state.clone()
            reward = self.getReward(life0, life1, self.ringDistance)
            complete = (life0*life1 == 0)
            current_value = F.sigmoid(self.value_model(outcome))
            continuation_value = reward if horizon == 0 else torch.where(complete, reward, current_value)
            continuation_prob = (1-p) ** (self.step_count+1) 
            value_target += continuation_value * continuation_prob
            return state, value_target

def get_simulation_state(simulation: Simulation)->Tensor:
    stateTensors = [
        simulation.agents[0].velocity,
        simulation.agents[0].position,
        simulation.blades[0].velocity,
        simulation.blades[0].position,
        simulation.agents[1].velocity,
        simulation.agents[1].position,
        simulation.blades[1].velocity,
        simulation.blades[1].position
    ]
    simulation_state = torch.cat(stateTensors,dim=1)
    return simulation_state

def get_random_directions(count: int)->Tensor:
    normals = torch.randn(count, 2)
    unit = F.normalize(normals,p=2,dim=1)
    return unit

def get_random_vectors(count: int, max_scale=1.0) ->Tensor:
    directions = get_random_directions(count)
    scales = max_scale*torch.rand(count).unsqueeze(1)
    return scales*directions

