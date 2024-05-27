from mesa import Agent
import random
from personality import Personality

class SocialAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.speed = random.random()
        self.personality = Personality(
            openness=random.random(),
            conscientiousness=random.random(),
            extraversion=random.random(),
            agreeableness=random.random(),
            neuroticism=random.random()
        )
        self.friends = []

    def step(self):
        # 基於個性和其他因素移動
        self.move()

        # 與其他代理互動
        self.interact_with_others()

    def move(self):
        # 基於個性和速度移動
        possible_steps = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
        new_position = self.random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)

    def interact_with_others(self):
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        for agent in cellmates:
            if agent != self:
                self.form_friendship(agent)

    def form_friendship(self, other_agent):
        # 根據性格形成友誼
        if other_agent not in self.friends:
            similarity = self.personality_similarity(other_agent)
            if similarity > 0.5:  # 假設相似度超過0.5會成為朋友
                self.friends.append(other_agent)
                other_agent.friends.append(self)

    def personality_similarity(self, other_agent):
        p1 = self.personality
        p2 = other_agent.personality
        return 1 - (abs(p1.openness - p2.openness) +
                    abs(p1.conscientiousness - p2.conscientiousness) +
                    abs(p1.extraversion - p2.extraversion) +
                    abs(p1.agreeableness - p2.agreeableness) +
                    abs(p1.neuroticism - p2.neuroticism)) / 5
