from math import pi
from numpy import where
import torch
from torch import Tensor, tensor
from torch.func import vmap, grad
import torch.nn.functional as F
from value import ValueModel, state_size
from simulation import Agent, Blade, Simulation, active_action_tensor, physics_dtype

unit_square = torch.tensor([[-1,-1],[1,-1],[1,1],[-1,1]]).to(physics_dtype)

class DataGenerator:
    def __init__(self, value_model: ValueModel, sim_count = 3, step_count=50):
        self.value_model = value_model
        self.get_costate = vmap(grad(lambda x: self.value_model(x).sum()))
        self.sim_count = sim_count
        self.step_count = step_count
        self.time_step = 0.1
        start_time = torch.arange(self.step_count)
        step_time = torch.arange(self.step_count)
        p = 0.005 # Discount Rate
        dt = step_time.unsqueeze(0) - start_time.unsqueeze(1)
        self.end_probs = torch.where(dt < 0, 0.0, p * (1 - p) ** dt)
        self.continuation_probs = (1-p) ** (self.step_count - start_time)
        self.simulation = Simulation(sim_count, self.time_step)
        self.agent0 = Agent(self.simulation, 0)
        self.blade0 = Blade(self.simulation, self.agent0)
        self.agent1 = Agent(self.simulation, 1)
        self.blade1 = Blade(self.simulation, self.agent1)
        self.scale: Tensor
        self.state: Tensor
        self.costate: Tensor
        self.vgrad0: Tensor
        self.vgrad1: Tensor
        self.gap0: Tensor
        self.gap1: Tensor
        self.life0: Tensor
        self.life1: Tensor
        self.centerDistance0: Tensor
        self.centerDistance1: Tensor
        self.reward: Tensor
        self.reset()
    
    def reset(self):
        n = self.sim_count
        z = torch.zeros(n, 2)
        features = [
            self.agent0.position, self.blade0.position,
            self.agent1.position, self.blade1.position,
            self.agent0.velocity, self.blade0.velocity,
            self.agent1.velocity, self.blade1.velocity,
        ]
        def static():
            vector = (20 + 10*torch.rand(n, 1)) * get_random_directions(n)
            return vector, vector, z, z, z, z, z, z
        def dynamic():
            a0p = get_random_vectors(n, 100)
            b0p = a0p + get_random_vectors(n, 80)
            a1p = get_random_vectors(n, 100)
            b1p = a1p + get_random_vectors(n, 80)
            a0v = get_random_vectors(n, 30)
            b0v = get_random_vectors(n, 70)
            a1v = get_random_vectors(n, 30)
            b1v = get_random_vectors(n, 70)
            return a0p, b0p, a1p, b1p, a0v, b0v, a1v, b1v
        def attack():
            direction = get_random_directions(n)
            a0_dist = 20 + 20*torch.rand(n, 1)
            b0_dist = a0_dist + 50*torch.rand(n, 1)
            a0p = a0_dist * direction
            b0p = b0_dist * direction + get_random_vectors(n, 50)
            return a0p, b0p, z, z, z, get_random_vectors(n, 70), z, z
        situations = [static(), dynamic(), attack()]
        choice = torch.randint(0, len(situations), (n, 1))
        for i, feature in enumerate(features):
            for j, situation in enumerate(situations):
                feature[:] = torch.where(choice == j, situation[i], feature)
        self.simulation.complete = torch.zeros((n, 1)).bool()

    def update(self):
        self.state = get_simulation_state(self.simulation)
        self.gap0 = torch.norm(self.agent0.position - self.blade1.position,p=2,dim=1,keepdim=True)
        self.gap1 = torch.norm(self.agent1.position - self.blade0.position,p=2,dim=1,keepdim=True)
        self.life0 = torch.where(self.gap0 > 15, 1, 0).to(physics_dtype)
        self.life1 = torch.where(self.gap1 > 15, 1, 0).to(physics_dtype)
        self.simulation.complete = self.life0 * self.life1 == 0
        self.centerDistance0 = torch.norm(self.agent0.position,p=2,dim=1,keepdim=True)
        self.centerDistance1 = torch.norm(self.agent1.position,p=2,dim=1,keepdim=True)
        ringSize0 = 20
        ringSize1 = 20
        ringOut0 = torch.where(self.centerDistance0 > ringSize0, 1, 0)
        ringOut1 = torch.where(self.centerDistance1 > ringSize1, 1, 0)
        self.reward = self.life0 * (0.01 + 0.99 * torch.maximum(1 - self.life1, ringOut1))

    def act(self, horizon: int):
        if horizon==0:
            self.costate = 0*self.state
            self.vgrad0 = +self.costate[:,[0,1]]
            self.agent0.action = torch.zeros(self.sim_count).int()
            self.agent1.action = torch.zeros(self.sim_count).int()
        else:
            self.costate = self.get_costate(self.state)
            self.vgrad0 = +self.costate[:,[0,1]]
            self.vgrad1 = -self.costate[:,[8,9]]
            action_values0 = torch.einsum('ij,kj->ik',self.vgrad0,active_action_tensor)
            action_values1 = torch.einsum('ij,kj->ik',self.vgrad1,active_action_tensor)
            self.agent0.action = torch.argmax(action_values0, dim=1)+1
            # self.agent1.action = torch.argmax(action_values1, dim=1)+1
            self.agent1.action = torch.zeros(self.sim_count).int()

    def generate(self, horizon: int)->tuple[Tensor,...]:
        self.value_model.eval()
        p = 0.005 # Discount Rate
        with torch.no_grad():
            state = torch.zeros((self.sim_count,self.step_count,state_size))
            target = torch.zeros((self.sim_count,self.step_count))
            self.reset()
            for t in range(self.step_count):
                self.simulation.step()
                self.update()
                self.act(horizon)
                state[:,t,:] = self.state
                end_prob = self.end_probs[:, t].view(1, self.step_count)
                target[:,:] += end_prob * self.reward
            continuation_value = self.reward if horizon == 0 else F.sigmoid(self.value_model(self.state))
            continuation_prob = self.continuation_probs.view(1, self.step_count)
            target[:,:] += continuation_prob * continuation_value
            target = torch.clamp(target, 0.0, 1.0)
            state = state.reshape(self.sim_count * self.step_count, state_size)
            target = target.reshape(self.sim_count * self.step_count, 1)
            return state, target

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

