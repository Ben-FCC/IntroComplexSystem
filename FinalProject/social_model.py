from mesa import Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from social_agent import SocialAgent
from mesa.datacollection import DataCollector

class SocialModel(Model):
    def __init__(self, N, width, height, alpha=0.1, max_speed=3.0, break_prob=0.05, phi=0.01):
        self.num_agents = N
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)
        self.alpha = alpha
        self.max_speed = max_speed
        self.break_prob = break_prob
        self.phi = phi

        # Create agents
        for i in range(self.num_agents):
            a = SocialAgent(i, self, alpha=self.alpha, max_speed=self.max_speed, break_prob=self.break_prob, phi=self.phi)
            self.schedule.add(a)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))

        self.datacollector = DataCollector(
            agent_reporters={"Position": "pos", "Friends": lambda a: len(a.friends)}
        )

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()
