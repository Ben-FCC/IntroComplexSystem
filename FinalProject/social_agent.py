from mesa import Agent
import numpy as np
import random
from personality import Personality

class SocialAgent(Agent):
    def __init__(self, unique_id, model, alpha=0.1):
        super().__init__(unique_id, model)
        self.pos = (random.randrange(model.grid.width), random.randrange(model.grid.height))
        self.speed = random.random()
        self.velocity = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])
        self.velocity = self.velocity / np.linalg.norm(self.velocity) * self.speed
        self.personality = Personality(
            openness=random.random(),
            conscientiousness=random.random(),
            extraversion=random.random(),
            agreeableness=random.random(),
            neuroticism=random.random()
        )
        self.friends = []
        self.alpha = alpha

    def step(self):
        # 基於個性和其他因素移動
        self.move()

        # 與其他代理互動
        self.interact_with_others()

    def move(self):
        # 基於速度方向移動
        new_position = np.array(self.pos) + self.velocity

        # 添加朋友位置的社會重力效應
        social_gravity_force = np.array([0.0, 0.0])
        for friend in self.friends:
            distance_vector = np.array(friend.pos) - np.array(self.pos)
            distance = np.linalg.norm(distance_vector)
            if distance > 0:  # 避免除以零
                social_gravity_force += self.alpha * self.personality.extraversion * distance_vector / (distance ** 2)
        
        self.velocity += social_gravity_force
        self.velocity = self.velocity / np.linalg.norm(self.velocity) * self.speed

        new_position = new_position + social_gravity_force
        new_position = np.mod(new_position, [self.model.grid.width, self.model.grid.height])
        new_position = (int(new_position[0]), int(new_position[1]))
        self.model.grid.move_agent(self, new_position)
        self.pos = new_position

        # 添加隨機變動以模擬真實運動
        angle_change = random.uniform(-np.pi / 4, np.pi / 4)
        rotation_matrix = np.array([[np.cos(angle_change), -np.sin(angle_change)],
                                    [np.sin(angle_change), np.cos(angle_change)]])
        self.velocity = np.dot(self.velocity, rotation_matrix)

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
