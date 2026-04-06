import os
import numpy as np
import torch
import torch.nn.functional as F
import arcade
from arcade import csscolor
from arcade.types import Point2List
from collections import defaultdict
from generator import DataGenerator
from models import ActionModel, ValueModel
import physics
from physics import Agent, Blade, actionVectors, get_simulation_state
from objective import get_reward, get_action_values

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
        self.set_update_rate(1 / 30)
        self.accumulator = 0
        self.generator = generator
        self.simulation = generator.start_simulation
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
        self.boundaryPolygons: list[Point2List] = []
        corner_count = len(self.simulation.boundary.corners)
        corners = [SCALE * self.simulation.boundary.corners[i][0,:] for i in range(corner_count)]
        polygon = tuple( (p[0].item(), p[1].item()) for p in corners)
        self.boundaryPolygons.append(polygon)


    def on_key_press(self, symbol: int, modifiers: int):
        self.pressed[symbol] = True

    def on_key_release(self, symbol: int, modifiers: int):
        self.pressed[symbol] = False

    def on_mouse_scroll(self, x: int, y: int, scroll_x: float, scroll_y: float):
       self.camera.zoom *= 1 + 0.1*scroll_y

    def on_draw(self):
        # fps = arcade.get_fps()
        # print(f"FPS: {fps:.2f}")
        self.clear()
        self.camera.use()
        i = self.index
        for boundaryPolygon in self.boundaryPolygons:
            arcade.draw_polygon_filled(boundaryPolygon, color=csscolor.BLACK)
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
            arcade.draw_line(x0,y0,x1,y1,circle._color,10)
        self.sprites.draw()

        # Graphically show the chosen action:
        # agent = self.agentCircles[0].agent
        # action_vector = actionVectors[agent.action[i]]
        # x0 = SCALE * agent.position[i,0].item()
        # y0 = SCALE * agent.position[i,1].item()
        # x1 = SCALE * (agent.position[i,0].item() + 6*action_vector[0].item())
        # y1 = SCALE * (agent.position[i,1].item() + 6*action_vector[1].item())
        # arcade.draw_line(x0,y0,x1,y1,csscolor.RED,5)

        # Graphically show the outcome state positions:
        # generator.generate_outcomes()
        # outcome_positions = self.generator.outcome_agent0.position
        # outcome_positions = outcome_positions.reshape(self.generator.batch_size,9,9,-1)
        # outcome_positions = outcome_positions[i,:,0,:]
        # start_position = self.generator.start_agent0.position[0]
        # x0 = SCALE * start_position[0].item()
        # y0 = SCALE * start_position[1].item()
        # for action in range(9):
        #     outcome_position = outcome_positions[action]
        #     x1 = SCALE * outcome_position[0].item()
        #     y1 = SCALE * outcome_position[1].item()
        #     arcade.draw_line(x0,y0,x1,y1,csscolor.WHITE,1)


    def on_update(self, delta_time: float) -> bool | None:
        self.agentCircles[1].agent.action[self.index] = self.get_user_action()
        self.simulation.step()
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
            dots = torch.einsum('ij,j->i',actionVectors, vector)
            action = torch.argmax(dots).item()
        return action

value_checkpoint_path = './checkpoints/value_checkpoint.pt'
action_checkpoint_path = './checkpoints/action_checkpoint.pt'
value_model = ValueModel()
action_model = ActionModel()

if os.path.exists(action_checkpoint_path):
    print('Loading Action Checkpoint...')
    checkpoint = torch.load(action_checkpoint_path, weights_only=False)
    action_model.load_state_dict(checkpoint['model_state_dict'])

if os.path.exists(value_checkpoint_path):
    print('Loading Value Checkpoint...')
    value_checkpoint = torch.load(value_checkpoint_path, weights_only=False)
    value_model.load_state_dict(value_checkpoint['model_state_dict'])

generator = DataGenerator(batch_size=5,timeStep=0.1,boundary_size=50)
generator.reset()

# TO DO:
# Setup the physics engine to let the boundary vary across batches.
# reward = change in objective

def action_callback():
    state = get_simulation_state(generator.start_simulation)
    action_logits = action_model(state)
    chosen_action = torch.argmax(action_logits, dim=1)
    generator.start_agent0.action = chosen_action


game = Game(generator,action_callback)
arcade.enable_timings()
game.run()