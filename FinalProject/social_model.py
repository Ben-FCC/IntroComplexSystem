from mesa import Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from social_agent import SocialAgent
from mesa.datacollection import DataCollector

class SocialModel(Model):
    def __init__(self, N, width, height, alpha=0.1):
        self.num_agents = N
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)
        self.alpha = alpha

        # Create agents
        for i in range(self.num_agents):
            a = SocialAgent(i, self, alpha=self.alpha)
            self.schedule.add(a)
            self.grid.place_agent(a, (self.random.randrange(self.grid.width), self.random.randrange(self.grid.height)))

        self.datacollector = DataCollector(
            agent_reporters={"Position": "pos", "Friends": lambda a: len(a.friends)}
        )

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()
