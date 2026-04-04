import os
import torch
import torch.nn.functional as F
import arcade
from arcade import csscolor
from arcade.types import Point2List
from collections import defaultdict
from generator import DataGenerator
from models import ActionModel
import physics
from physics import Agent, Blade, Simulation, actionVectors, visionCast
from reward import get_reward

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

action_checkpoint_path = './checkpoints/action_checkpoint.pt'
action_model = ActionModel()

if os.path.exists(action_checkpoint_path):
    print('Loading Action Checkpoint...')
    checkpoint = torch.load(action_checkpoint_path, weights_only=False)
    action_model.load_state_dict(checkpoint['model_state_dict'])

generator = DataGenerator(count=3, timeStep=0.1)
generator.setup()
simulation = generator.simulation

def action_callback():
    vision0 = visionCast(generator.agent0.position, generator.visionReach, generator.simulation.boundary.walls)
    stateTensors = [
        generator.agent1.position - generator.agent0.position,
        generator.agent1.velocity,
        generator.blade1.position - generator.agent1.position,
        generator.blade1.velocity,
        generator.agent0.velocity,
        vision0,
    ]
    state = torch.cat(stateTensors,dim=1)
    vision = state[:,-8:]
    print(vision[0,:].to(torch.int).detach().cpu().tolist())
    action_logits = action_model(state)
    action_probs = F.softmax(action_logits,1)
    action = torch.multinomial(action_probs,1).squeeze(1)
    generator.agent0.action = action

game = Game(simulation,action_callback)
arcade.enable_timings()
game.run()