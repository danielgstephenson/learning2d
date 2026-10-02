import torch
from torch import Tensor
import arcade
from arcade import csscolor
from arcade.types import Color
from collections import defaultdict

from generator import DataGenerator
from physics import Agent, Blade, action_tensor, device, time_step

SCALE = 10

torch.set_default_device(device)

class AgentCircle(arcade.SpriteCircle):
    def __init__(self, index: int, agent: Agent):
        radius = SCALE * agent.radius
        color = csscolor.GREEN
        if agent.align == 1: color = csscolor.BLUE
        x = agent.position[index,0].item()
        y = agent.position[index,1].item()
        super().__init__(radius, color, False, x, y)
        self.agent = agent

class BladeCircle(arcade.SpriteCircle):
    def __init__(self, index: int, blade: Blade):
        radius = SCALE * blade.radius
        color = (0,255,50,255)
        if blade.agent.align == 1: color = csscolor.AQUA
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
        self.set_update_rate(1 / 50)
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
        self.bot_action = 0
        self.frame_counter = 0
        self.state: Tensor

    def on_key_press(self, symbol: int, modifiers: int):
        self.pressed[symbol] = True
        if symbol == arcade.key.ENTER:
            self.gen.reset()
            self.frame_counter = 0
            self.paused = True

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
        time = self.world.time[0].item()
        text = f'Time: {time:.1f}, '
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
        arcade.draw_circle_outline(0, 0, SCALE*50, arcade.color.GRAY,SCALE*1)
        for circle in self.bladeCircles:
            circle.center_x = SCALE * circle.blade.position[self.index,0].item()
            circle.center_y = SCALE * circle.blade.position[self.index,1].item()
        for circle in self.agentCircles:
            circle.center_x = SCALE * circle.agent.position[self.index,0].item()
            circle.center_y = SCALE * circle.agent.position[self.index,1].item()
        for circle in self.bladeCircles:
            self.draw_line(circle.blade.position, circle.blade.agent.position, circle.color,10)
        self.sprites.draw()
        self.draw_text()

    def on_update(self, delta_time: float) -> bool | None:
        self.camera.position = self.agentCircles[1].position
        # self.camera.position = (0,0)
        if self.paused: return
        ongoing = self.gen.agent0.alive & self.gen.agent1.alive
        dt = torch.where(ongoing, time_step, 0)
        self.world.step(dt)
        self.gen.update()
        self.gen.agent1.action[self.index] = self.get_user_action()
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
        
gen = DataGenerator(batch_size=1)
gen.model.noise = 0.0
stage = 0

game = Game(gen)
arcade.enable_timings()
game.run()