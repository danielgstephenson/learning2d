import os
import torch
from torch import Tensor
import torch.nn.functional as F
import numpy as np
import csv
import arcade
from arcade import csscolor
from arcade.types import Point2List, Color
from collections import defaultdict
from torch.func import vmap, grad
import onnxruntime as ort

from generator import DataGenerator
from models import ValueModel
import world as world
from world import Agent, Blade, action_tensor

SCALE = 10

torch.set_default_device(world.device)

class AgentCircle(arcade.SpriteCircle):
    def __init__(self, index: int, agent: Agent):
        radius = SCALE * agent.radius
        color = csscolor.GREEN
        if agent.align == 1: color = csscolor.BLUE
        if agent.align == 2: color = csscolor.RED
        x = agent.position[index,0].item()
        y = agent.position[index,1].item()
        super().__init__(radius, color, False, x, y)
        self.agent = agent

class BladeCircle(arcade.SpriteCircle):
    def __init__(self, index: int, blade: Blade):
        radius = SCALE * blade.radius
        color = (100,255,50,255)
        if blade.agent.align == 1: color = csscolor.AQUA
        if blade.agent.align == 2: color = csscolor.MAGENTA
        x = blade.position[index,0].item()
        y = blade.position[index,1].item()
        super().__init__(radius, color, False, x, y)
        self.alpha = 100
        self.blade = blade

class Game(arcade.Window):
    def __init__(self, gen: DataGenerator):
        window_size = 900
        super().__init__(window_size, window_size, 'learning2d')
        arcade.set_background_color((0,0,0,255))
        self.camera = arcade.Camera2D()
        self.camera.zoom = 0.1
        self.hud_camera = arcade.Camera2D()
        self.hud_camera.position = (0,0)
        self.index = 0
        self.set_update_rate(1 / 40)
        self.gen = gen
        self.world = gen.world
        self.pressed = defaultdict(lambda: False)
        self.agentCircles: list[AgentCircle] = []
        self.bladeCircles: list[BladeCircle] = []
        self.sprites = arcade.SpriteList()
        self.paused = True
        for blade in self.world.blades:
            blade_circle = BladeCircle(self.index, blade)
            self.bladeCircles.append(blade_circle)
            self.sprites.append(blade_circle)
        for blade in self.world.agents:
            agent_circle = AgentCircle(self.index, blade)
            self.agentCircles.append(agent_circle)
            self.sprites.append(agent_circle)
        self.value_estimate = 0
        self.velocity_gradient = [0, 0]
        self.agent_action = 0
        self.reset_log_file()
        self.frame_counter = 0
        self.state: Tensor

    def on_key_press(self, symbol: int, modifiers: int):
        self.pressed[symbol] = True
        if symbol == arcade.key.ENTER:
            self.gen.reset()
            self.frame_counter = 0
            self.paused = True
            self.reset_log_file()
        if symbol == arcade.key.L:
            self.gen.reset_custom()
            self.frame_counter = 0
            self.paused = True
            self.reset_log_file()

    def on_key_release(self, symbol: int, modifiers: int):
        self.pressed[symbol] = False
        if symbol == arcade.key.SPACE:
            self.paused = not self.paused

    def on_mouse_scroll(self, x: int, y: int, scroll_x: float, scroll_y: float):
       self.camera.zoom *= 1 + 0.1*scroll_y

    def draw_line(self, start, end, color: Color, width: int | float):
        x0 = SCALE * start[self.index,0].item()
        y0 = SCALE * start[self.index,1].item()
        x1 = SCALE * end[self.index,0].item()
        y1 = SCALE * end[self.index,1].item()
        arcade.draw_line(x0,y0,x1,y1,color,width)

    def draw_point(self, point, radius: int | float, color: Color):
        x = SCALE * point[self.index,0].item()
        y = SCALE * point[self.index,1].item()
        arcade.draw_circle_filled(x,y,radius,color)

    def draw_text(self):
        self.hud_camera.use()
        text = f'Time: {self.world.time:.1f}, '
        text += f'FPS: {arcade.get_fps():.1f}, '
        text += f'Reward: {self.gen.reward[self.index].item():0.3f}'
        x = 0
        y = 400
        color = arcade.color.WHITE
        font_size = 16
        arcade.draw_text(text,x,y,color,font_size,anchor_x="center")
        self.camera.use()

    def on_draw(self):
        self.clear()
        self.camera.use()
        arcade.draw_circle_outline(0, 0, SCALE*gen.ring_size, arcade.color.GRAY,SCALE*1)
        # arcade.draw_arc_outline(0,0,SCALE*35,SCALE*35,arcade.color.GRAY,0,360*charge,SCALE*2)
        for circle in self.bladeCircles:
            circle.center_x = SCALE * circle.blade.position[self.index,0].item()
            circle.center_y = SCALE * circle.blade.position[self.index,1].item()
        for circle in self.agentCircles:
            circle.center_x = SCALE * circle.agent.position[self.index,0].item()
            circle.center_y = SCALE * circle.agent.position[self.index,1].item()
            circle.alpha = 255 if circle.agent.alive[self.index,0].item() else 0
        for circle in self.bladeCircles:
            if circle.blade.agent.alive[self.index,0].item():
                self.draw_line(circle.blade.position, circle.blade.agent.position, circle._color,10)
        self.sprites.draw()
        self.draw_text()

    def on_update(self, delta_time: float) -> bool | None:
        self.camera.position = self.agentCircles[1].position
        # self.camera.position = (0,0)
        if self.paused: return
        self.world.step()
        self.gen.update()
        agentPosition0 = self.world.agents[0].position[self.index,:]
        agentVelocity0 = self.world.agents[0].velocity[self.index,:]
        bladePosition0 = self.world.blades[0].position[self.index,:]
        bladeVelocity0 = self.world.blades[0].velocity[self.index,:]
        agentPosition1 = self.world.agents[1].position[self.index,:]
        agentVelocity1 = self.world.agents[1].velocity[self.index,:]
        bladePosition1 = self.world.blades[1].position[self.index,:]
        bladeVelocity1 = self.world.blades[1].velocity[self.index,:]
        state = self.gen.get_state()
        value_estimate = torch.sigmoid(gen.model(state))
        # state_np = state.cpu().numpy()
        # vgrad0 = torch.tensor(session.run(['grad'], {'state': state_np})[0])
        # vgrad1 = torch.tensor([[0.0,0.0]])
        # vgrads = (vgrad0,vgrad1)
        actions = gen.model.actions(state)
        gen.agent0.action = actions[0]
        # gen.agent1.action = actions[1]
        gen.agent1.action[self.index] = self.get_user_action()
        row = [
            stage,self.frame_counter+1,self.world.time,
            self.gen.agent0.alive[self.index,0].int().item(),
            self.gen.agent1.alive[self.index,0].int().item(),
            agentPosition0[0].detach().item(), agentPosition0[1].detach().item(), 
            agentVelocity0[0].detach().item(), agentVelocity0[1].detach().item(),
            bladePosition0[0].detach().item(), bladePosition0[1].detach().item(), 
            bladeVelocity0[0].detach().item(), bladeVelocity0[1].detach().item(),
            agentPosition1[0].detach().item(), agentPosition1[1].detach().item(), 
            agentVelocity1[0].detach().item(), agentVelocity1[1].detach().item(),
            bladePosition1[0].detach().item(), bladePosition1[1].detach().item(), 
            bladeVelocity1[0].detach().item(), bladeVelocity1[1].detach().item(),
            gen.reward[self.index].detach().item(),
            value_estimate[self.index,0].detach().item(),
            gen.agent0.action[self.index,0].detach().item(),
            gen.agent1.action[self.index,0].detach().item(),
        ]
        self.log_writer.writerow(row)
        self.log_file.flush()
        self.frame_counter += 1

    def reset_log_file(self):
        self.log_file = open("./simulation/simulation.csv", mode='w', newline="")
        self.log_writer = csv.writer(self.log_file)
        self.log_writer.writerow([
            "stage","frame","time","life0","life1",
            "a0x","a0y","a0vx","a0vy",
            "b0x","b0y","b0vx","b0vy",
            "a1x","a1y","a1vx","a1vy",
            "b1x","b1y","b1vx","b1vy",
            "reward","value",
            "action0","action1",
        ])

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
        

checkpoint_path = './checkpoints/checkpoint.pt'
# session = ort.InferenceSession('./onnx/guard.onnx')
gen = DataGenerator(batch_size=1)
gen.model.noise = 0.0
stage = 0

if os.path.exists(checkpoint_path):
    print(f'Loading Checkpoint from {checkpoint_path}...')
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    gen.model.load_state_dict(checkpoint['gen_model'])
    stage = checkpoint['stage']

game = Game(gen)
arcade.enable_timings()
game.run()