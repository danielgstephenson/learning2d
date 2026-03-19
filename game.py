
import torch
import arcade
from arcade import csscolor
import physics
from physics import Agent, Simulation

SCALE = 10
AGENT_RADIUS = 5 * SCALE

torch.set_default_device(physics.device)

simulation = Simulation(2)
agent0 = Agent(simulation, 0)
agent1 = Agent(simulation, 1)
agent2 = Agent(simulation, 2)
agent0.velocity = 20*(1 - 2*torch.rand((simulation.count,2)))
agent1.position = torch.rand((simulation.count,2))
agent2.position = torch.rand((simulation.count,2))

class Game(arcade.Window):
    def __init__(self):
        super().__init__(800, 600, 'learning2d')
        arcade.set_background_color(csscolor.BLACK)
        self.camera = arcade.Camera2D()
        self.index = 0
        self.sprites = arcade.SpriteList()
        self.circle0 = arcade.SpriteCircle(AGENT_RADIUS, arcade.csscolor.BLUE)
        self.sprites.append(self.circle0)
        self.circle1 = arcade.SpriteCircle(AGENT_RADIUS, arcade.csscolor.GREEN)
        self.sprites.append(self.circle1)
        self.circle2 = arcade.SpriteCircle(AGENT_RADIUS, arcade.csscolor.GREEN)
        self.sprites.append(self.circle2)

    def on_draw(self):
        self.clear()
        self.camera.use()
        i = self.index
        self.circle0.center_x = SCALE * agent0.position[i,0].item()
        self.circle0.center_y = SCALE * agent0.position[i,1].item()
        self.circle1.center_x = SCALE * agent1.position[i,0].item()
        self.circle1.center_y = SCALE * agent1.position[i,1].item()
        self.circle2.center_x = SCALE * agent2.position[i,0].item()
        self.circle2.center_y = SCALE * agent2.position[i,1].item()
        self.sprites.draw()

    def on_update(self, delta_time: float) -> bool | None:
        self.camera.position = self.circle0.position
        simulation.step()
    

game = Game()
game.run()

# arcade.open_window(800, 800, 'learning2d')
# arcade.set_background_color(arcade.csscolor.BLACK)
# arcade.start_render()
# arcade.draw_circle_filled(100,100,20,arcade.csscolor.BLUE)
# arcade.finish_render()
# arcade.run()