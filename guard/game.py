import os
import torch
import torch.nn.functional as F
import onnxruntime as ort
import numpy as np
import csv
from datetime import datetime
import arcade
from arcade import csscolor
from arcade.types import Point2List, Color
from collections import defaultdict
from generator import DataGenerator
from value import ValueModel
import world as world
from torch.func import vmap, grad
from world import Agent, Blade, device, action_tensor, active_action_tensor
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
    def __init__(self, generator: DataGenerator):
        super().__init__(1000, 1000, 'learning2d')
        arcade.set_background_color((20,20,20,255))
        self.camera = arcade.Camera2D()
        self.camera.zoom = 0.1
        self.index = 0
        self.set_update_rate(1 / 40)
        self.generator = generator
        self.world = generator.world
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

    def reset_log_file(self):
        self.log_file = open("./simulation/simulation.csv", mode='w', newline="")
        self.log_writer = csv.writer(self.log_file)
        self.log_writer.writerow([
            "frame","time","life0","life1","charge",
            "a0_x", "a0_y", "a0_vx", "a0_vy",
            "b0_x", "b0_y", "b0_vx", "b0_vy",
            "a1_x", "a1_y", "a1_vx", "a1_vy",
            "b1_x", "b1_y", "b1_vx", "b1_vy",
            "grad_a0_vx", "grad_a0_vy",
            "grad_a1_vx", "grad_a1_vy",
            "reward", "value_estimate",
            "action0", "action1",
            'c0x','c0y','c1x','c1y','c2x','c2y','c3x','c3y'
        ])

    def on_key_press(self, symbol: int, modifiers: int):
        self.pressed[symbol] = True
        if symbol == arcade.key.SPACE:
            self.paused = not self.paused
        if symbol == arcade.key.ENTER:
            self.generator.reset()
            self.frame_counter = 0
            self.paused = True
            self.reset_log_file()
        if symbol == arcade.key.L:
            self.generator.reset_custom()
            self.frame_counter = 0
            self.paused = True
            self.reset_log_file()

    def on_key_release(self, symbol: int, modifiers: int):
        self.pressed[symbol] = False

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

    def on_draw(self):
        self.clear()
        self.camera.use()
        corner_count = self.world.boundary.num_walls
        corners = [SCALE * self.world.boundary.wall_starts[self.index,i,:] for i in range(corner_count)]
        self.boundaryPolygon: Point2List = tuple( (p[0].item(), p[1].item()) for p in corners)
        arcade.draw_polygon_filled(self.boundaryPolygon, color=csscolor.BLACK)
        arcade.draw_circle_outline(0, 0, SCALE*13, arcade.color.GRAY, SCALE*1)
        charge = self.generator.world.charge[self.index].item()
        start_angle = 270
        end_angle = 270 + 360 * charge
        arcade.draw_arc_outline(0, 0, SCALE*30, SCALE*30, arcade.color.GRAY, start_angle, end_angle, border_width=SCALE*2)
        text = f'FPS: {arcade.get_fps():.1f}, Time: {self.world.time:.1f}'
        arcade.draw_text(text,x=SCALE*0,y=SCALE*150,color=arcade.color.WHITE,font_size=SCALE*16)
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

    def on_update(self, delta_time: float) -> bool | None:
        self.camera.position = self.agentCircles[1].position
        if self.paused: return
        if self.world.charge[self.index,0].item() >= 1: return
        self.world.step()
        self.generator.update()
        agentPosition0 = self.world.agents[0].position[self.index,:]
        agentVelocity0 = self.world.agents[0].velocity[self.index,:]
        bladePosition0 = self.world.blades[0].position[self.index,:]
        bladeVelocity0 = self.world.blades[0].velocity[self.index,:]
        agentPosition1 = self.world.agents[1].position[self.index,:]
        agentVelocity1 = self.world.agents[1].velocity[self.index,:]
        bladePosition1 = self.world.blades[1].position[self.index,:]
        bladeVelocity1 = self.world.blades[1].velocity[self.index,:]
        state = self.generator.get_simulation_state()
        value_estimate = value_model(state)
        costate = get_costate(state)
        velocity_grad0 = +costate[:,[0,1]]
        velocity_grad1 = -costate[:,[8,9]]
        action_values0 = torch.einsum('ij,kj->ik',velocity_grad0,active_action_tensor)
        action_values1 = torch.einsum('ij,kj->ik',velocity_grad1,active_action_tensor)
        generator.agent0.action = torch.argmax(action_values0, dim=1) + 1
        generator.agent1.action = torch.argmax(action_values1, dim=1) + 1
        # self.agentCircles[1].agent.action[self.index] = self.get_user_action()
        row = [
            self.frame_counter+1,self.world.time,
            self.generator.agent0.alive[self.index,0].int().item(),
            self.generator.agent1.alive[self.index,0].int().item(),
            self.world.charge[self.index,0].item(),
            agentPosition0[0].detach().item(), agentPosition0[1].detach().item(), 
            agentVelocity0[0].detach().item(), agentVelocity0[1].detach().item(),
            bladePosition0[0].detach().item(), bladePosition0[1].detach().item(), 
            bladeVelocity0[0].detach().item(), bladeVelocity0[1].detach().item(),
            agentPosition1[0].detach().item(), agentPosition1[1].detach().item(), 
            agentVelocity1[0].detach().item(), agentVelocity1[1].detach().item(),
            bladePosition1[0].detach().item(), bladePosition1[1].detach().item(), 
            bladeVelocity1[0].detach().item(), bladeVelocity1[1].detach().item(),
            velocity_grad0[self.index,0].detach().item(), velocity_grad0[self.index,1].detach().item(),
            velocity_grad1[self.index,0].detach().item(), velocity_grad1[self.index,1].detach().item(),
            generator.reward[self.index].detach().item(),
            value_estimate[self.index,0].detach().item(),
            generator.agent0.action[self.index].detach().item(),
            generator.agent1.action[self.index].detach().item()
        ]
        c0 = self.world.boundary.wall_starts[self.index,0,:].detach()
        row += [c0[0].item(),c0[1].item()]
        c1 = self.world.boundary.wall_starts[self.index,1,:].detach()
        row += [c1[0].item(),c1[1].item()]
        c2 = self.world.boundary.wall_starts[self.index,2,:].detach()
        row += [c2[0].item(),c2[1].item()]
        c3 = self.world.boundary.wall_starts[self.index,3,:].detach()
        row += [c3[0].item(),c3[1].item()]
        self.log_writer.writerow(row)
        self.log_file.flush()
        self.frame_counter += 1

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
value_model = ValueModel()
value_model.eval()

if os.path.exists(checkpoint_path):
    print('Loading Value Checkpoint...')
    value_checkpoint = torch.load(checkpoint_path, weights_only=False)
    value_model.load_state_dict(value_checkpoint['model_state_dict'])

generator = DataGenerator(value_model,world_count=1,time_step=0.04)

get_costate = vmap(grad(lambda x: value_model(x).sum()))
    
game = Game(generator)
arcade.enable_timings()
game.run()