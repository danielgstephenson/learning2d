import os
from matplotlib.backend_bases import key_press_handler
import numpy as np
import torch
import torch.nn.functional as F
import arcade
from arcade import csscolor
from arcade.types import Point2List
from collections import defaultdict
from generator import DataGenerator, get_simulation_state
from models import ActionModel, ValueModel
import physics
from physics import Agent, Blade, action_tensor
from objective import get_action_values
SCALE = 10

torch.set_default_device(physics.device)

class AgentCircle(arcade.SpriteCircle):
    def __init__(self, index: int, agent: Agent):
        radius = SCALE * agent.radius
        color = csscolor.GREEN
        if agent.align == 1: color = csscolor.BLUE
        if agent.align == 2: color = csscolor.RED
        x = agent.position[index,0].item()
        y = agent.position[index,0].item()
        super().__init__(radius, color, False, x, y)
        self.agent = agent

class BladeCircle(arcade.SpriteCircle):
    def __init__(self, index: int, blade: Blade):
        radius = SCALE * blade.radius
        color = (100,255,50,255)
        if blade.agent.align == 1: color = csscolor.AQUA
        if blade.agent.align == 2: color = csscolor.MAGENTA
        x = blade.position[index,0].item()
        y = blade.position[index,0].item()
        super().__init__(radius, color, False, x, y)
        self.blade = blade

class Game(arcade.Window):
    def __init__(self, generator: DataGenerator, update_callback = (lambda: None) ):
        super().__init__(800, 600, 'learning2d')
        arcade.set_background_color((20,20,20,255))
        self.update_callback = update_callback
        self.camera = arcade.Camera2D()
        self.camera.zoom = 0.1
        self.index = 0
        self.timeScale = 2
        self.set_update_rate(1 / 10)
        self.accumulator = 0
        self.generator = generator
        self.simulation = generator.simulation
        self.pressed = defaultdict(lambda: False)
        self.agentCircles: list[AgentCircle] = []
        self.bladeCircles: list[BladeCircle] = []
        self.sprites = arcade.SpriteList()
        for blade in self.simulation.blades:
            blade_circle = BladeCircle(self.index, blade)
            self.bladeCircles.append(blade_circle)
            self.sprites.append(blade_circle)
        for blade in self.simulation.agents:
            agent_circle = AgentCircle(self.index, blade)
            self.agentCircles.append(agent_circle)
            self.sprites.append(agent_circle)
        corner_count = len(self.simulation.boundary.corners)
        corners = [SCALE * self.simulation.boundary.corners[i][self.index,:] for i in range(corner_count)]
        self.boundaryPolygon: Point2List = tuple( (p[0].item(), p[1].item()) for p in corners)


    def on_key_press(self, symbol: int, modifiers: int):
        self.pressed[symbol] = True
        if symbol == arcade.key.SPACE:
            self.index = (self.index + 1) % self.generator.batch_size
            print('index',self.index)

    def on_key_release(self, symbol: int, modifiers: int):
        self.pressed[symbol] = False

    def on_mouse_scroll(self, x: int, y: int, scroll_x: float, scroll_y: float):
       self.camera.zoom *= 1 + 0.1*scroll_y

    def on_draw(self):
        self.clear()
        self.camera.use()
        corner_count = len(self.simulation.boundary.corners)
        corners = [SCALE * self.simulation.boundary.corners[i][self.index,:] for i in range(corner_count)]
        self.boundaryPolygon = tuple( (p[0].item(), p[1].item()) for p in corners)
        arcade.draw_polygon_filled(self.boundaryPolygon, color=csscolor.BLACK)
        for circle in self.bladeCircles:
            circle.center_x = SCALE * circle.blade.position[self.index,0].item()
            circle.center_y = SCALE * circle.blade.position[self.index,1].item()
        for circle in self.agentCircles:
            circle.center_x = SCALE * circle.agent.position[self.index,0].item()
            circle.center_y = SCALE * circle.agent.position[self.index,1].item()
        for circle in self.bladeCircles:
            x0 = SCALE * circle.blade.position[self.index,0].item()
            y0 = SCALE * circle.blade.position[self.index,1].item()
            x1 = SCALE * circle.blade.agent.position[self.index,0].item()
            y1 = SCALE * circle.blade.agent.position[self.index,1].item()
            arcade.draw_line(x0,y0,x1,y1,circle._color,10)
        self.sprites.draw()

    def on_update(self, delta_time: float) -> bool | None:
        self.agentCircles[1].agent.action[self.index] = self.get_user_action()
        agentPosition = self.simulation.agents[0].position[self.index,:]
        bladePosition = self.simulation.blades[0].position[self.index,:]
        distance = torch.norm(agentPosition-bladePosition,p=2,dim=0)
        if distance > 15: self.simulation.step()
        self.update_callback()
        self.camera.position = self.agentCircles[1].position

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
            dots = torch.einsum('ij,j->i',action_tensor, vector)
            action = torch.argmax(dots).item()
        return action
        

value_checkpoint_path = './checkpoints/value_checkpoint.pt'
action_checkpoint_path = './checkpoints/action_checkpoint.pt'
value_model = ValueModel()
action_model = ActionModel()

if os.path.exists(action_checkpoint_path):
    print('Loading Action 0 Checkpoint...')
    checkpoint = torch.load(action_checkpoint_path, weights_only=False)
    action_model.load_state_dict(checkpoint['model_state_dict'])

if os.path.exists(value_checkpoint_path):
    print('Loading Value Checkpoint...')
    value_checkpoint = torch.load(value_checkpoint_path, weights_only=False)
    value_model.load_state_dict(value_checkpoint['model_state_dict'])

generator = DataGenerator(batch_size=5,time_step=0.1)
generator.reset()

# TO DO:
# Setup the physics engine to let the boundary vary across batches.
# reward = change in objective

def action_callback():
    state = get_simulation_state(generator.simulation)
    value = value_model(state)
    print('value',value[game.index].item())
    action_logits = action_model(state)
    chosen_action = torch.argmax(action_logits, dim=1)
    generator.agent0.action = chosen_action


game = Game(generator,action_callback)
arcade.enable_timings()
game.run()