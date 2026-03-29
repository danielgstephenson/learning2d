import numpy as np
from math import pi, sqrt
from typing import Callable
import torch
from torch import Tensor
import arcade
from arcade import SpriteCircle, csscolor
from arcade.types import Point2List
from collections import defaultdict
from generator import DataGenerator
import physics
from physics import Agent, Blade, Boundary, Simulation, actionVectors, visionDirList, rayCastSegments, visionCast

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
        if blade.agent.align == 1: color = (100,255,50,255)
        if blade.agent.align == 2: color = csscolor.MAGENTA
        x = blade.position[index,0].item()
        y = blade.position[index,0].item()
        super().__init__(radius, color, False, x, y)
        self.blade = blade

class Game(arcade.Window):
    def __init__(self, simulation: Simulation, update_callback = (lambda: None) ):
        super().__init__(800, 600, 'learning2d')
        arcade.set_background_color((20,20,20,255))
        self.update_callback = update_callback
        self.camera = arcade.Camera2D()
        self.camera.zoom = 0.1
        self.index = 0
        self.timeScale = 2
        self.set_update_rate(1 / 30)
        self.accumulator = 0
        self.simulation = simulation
        self.pressed = defaultdict(lambda: False)
        self.agentCircles: list[AgentCircle] = []
        self.bladeCircles: list[BladeCircle] = []
        self.sprites = arcade.SpriteList()
        for blade in simulation.blades:
            blade_circle = BladeCircle(self.index, blade)
            self.bladeCircles.append(blade_circle)
            self.sprites.append(blade_circle)
        for blade in simulation.agents:
            agent_circle = AgentCircle(self.index, blade)
            self.agentCircles.append(agent_circle)
            self.sprites.append(agent_circle)
        self.boundaryPolygons: list[Point2List] = []
        polygon = tuple((SCALE*p[0], SCALE*p[1]) for p in self.simulation.boundary.points)
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
        # test segment cast
        # circle0 = self.agentCircles[0]
        # reach = 100
        # x0 = circle0.center_x
        # y0 = circle0.center_y
        # rayFactorMatrix = visionCast(circle0.agent.position,reach,self.simulation)
        # for i in range(8):
        #     a = visionDirList[i]
        #     x1 = x0 + SCALE * reach * a[0]
        #     y1 = y0 + SCALE * reach * a[1]
        #     white = (255, 255, 255, 50)
        #     red = (255, 0, 0, 100)
        #     color = white if rayFactorMatrix[self.index,i] > reach else red
        #     arcade.draw_line(x0, y0, x1, y1, color, 20)

    def on_update(self, delta_time: float) -> bool | None:
        self.agentCircles[0].agent.action[self.index] = self.get_user_action()
        self.simulation.step()
        self.update_callback()
        self.camera.position = self.agentCircles[0].position
    
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

generator = DataGenerator(count=3, timeStep=0.1)
generator.setup()
simulation = generator.simulation

game = Game(simulation)
arcade.enable_timings()
game.run()