from math import pi, sqrt
import torch
from torch import Tensor
import arcade
from arcade import SpriteCircle, csscolor
from collections import defaultdict
import physics
from physics import Agent, Blade, Simulation

SCALE = 10

torch.set_default_device(physics.device)

class AgentCircle(arcade.SpriteCircle):
    def __init__(self, index: int, agent: Agent):
        radius = SCALE * agent.radius
        color = csscolor.BLUE
        if agent.align == 1: color = csscolor.GREEN
        if agent.align == 2: color = csscolor.RED
        x = agent.position[index,0].item()
        y = agent.position[index,0].item()
        super().__init__(radius, color, False, x, y)
        self.agent = agent

class BladeCircle(arcade.SpriteCircle):
    def __init__(self, index: int, blade: Blade):
        radius = SCALE * blade.radius
        color = csscolor.AQUA
        if blade.agent.align == 1: color = csscolor.LIGHT_GREEN
        if blade.agent.align == 2: color = csscolor.MAGENTA
        x = blade.position[index,0].item()
        y = blade.position[index,0].item()
        super().__init__(radius, color, False, x, y)
        self.blade = blade


class Game(arcade.Window):
    def __init__(self, simulation: Simulation):
        super().__init__(800, 600, 'learning2d')
        arcade.set_background_color(csscolor.BLACK)
        self.camera = arcade.Camera2D()
        self.camera.zoom = 0.5
        self.index = 0
        self.timeScale = 2
        self.set_update_rate(simulation.timeStep / self.timeScale)
        self.simulation = simulation
        self.agentCircles: list[AgentCircle] = []
        self.bladeCircles: list[BladeCircle] = []
        self.sprites = arcade.SpriteList()
        self.pressed = defaultdict(lambda: False)
        for blade in simulation.blades:
            blade_circle = BladeCircle(self.index, blade)
            self.bladeCircles.append(blade_circle)
            self.sprites.append(blade_circle)
        for blade in simulation.agents:
            agent_circle = AgentCircle(self.index, blade)
            self.agentCircles.append(agent_circle)
            self.sprites.append(agent_circle)

    def on_key_press(self, symbol: int, modifiers: int):
        self.pressed[symbol] = True

    def on_key_release(self, symbol: int, modifiers: int):
        self.pressed[symbol] = False

    def on_mouse_scroll(self, x: int, y: int, scroll_x: float, scroll_y: float):
       self.camera.zoom *= 1 + 0.1*scroll_y

    def on_draw(self):
        self.clear()
        self.camera.use()
        i = self.index
        for circle in self.bladeCircles:
            circle.center_x = SCALE * circle.blade.position[i,0].item()
            circle.center_y = SCALE * circle.blade.position[i,1].item()
        for circle in self.agentCircles:
            circle.center_x = SCALE * circle.agent.position[i,0].item()
            circle.center_y = SCALE * circle.agent.position[i,1].item()
        for circle in self.bladeCircles:
            x0 = SCALE * circle.blade.position[i,0].item()
            y0 = SCALE * circle.blade.position[i,1].item()
            x1 = SCALE * circle.blade.agent.position[i,0].item()
            y1 = SCALE * circle.blade.agent.position[i,1].item()
            arcade.draw_line(x0,y0,x1,y1,circle._color,5)
        self.sprites.draw()
        arcade.draw_arc_filled(0,0,2*SCALE,2*SCALE,csscolor.RED,0,360) # DRAW TEST POINT

    def on_update(self, delta_time: float) -> bool | None:
        self.agentCircles[0].agent.action[self.index] = self.get_user_action()
        self.camera.position = self.agentCircles[0].position
        simulation.step()
    
    def get_user_action(self):
        dx = 0.0
        dy = 0.0
        if self.pressed[arcade.key.W] or self.pressed[arcade.key.UP]:
            dy += 1
        if self.pressed[arcade.key.S] or self.pressed[arcade.key.DOWN]:
            dy -= 1
        if self.pressed[arcade.key.A] or self.pressed[arcade.key.LEFT]:
            dx -= 1
        if self.pressed[arcade.key.D] or self.pressed[arcade.key.RIGHT]:
            dx += 1
        action = 0
        if dx != 0.0 or dy != 0.0:
            vector = torch.tensor([dx,dy])
            dots = torch.einsum('ij,j->i',self.simulation.action_tensor, vector)
            action = torch.argmax(dots).item()
        return action
        
simulation = Simulation(2)
agent0 = Agent(simulation, 0)
agent1 = Agent(simulation, 1)
agent2 = Agent(simulation, 1)
blade0 = Blade(simulation, agent0)
agent0.velocity = 20*(2*torch.rand((simulation.count,2)) - 1)
agent1.position = 4*torch.rand((simulation.count,2)) - 2
agent2.position = 4*torch.rand((simulation.count,2)) - 2
game = Game(simulation)
game.run()