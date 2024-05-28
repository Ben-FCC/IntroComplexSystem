from mesa import Agent
import numpy as np
import random
from personality import Personality

class SocialAgent(Agent):
    def __init__(self, unique_id, model, alpha=0.1, max_speed=3.0):
        super().__init__(unique_id, model)
        self.pos = (random.randrange(model.grid.width), random.randrange(model.grid.height))
        self.speed = random.random()
        self.velocity = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])
        self.velocity = self.velocity / np.linalg.norm(self.velocity) * self.speed

        # 生成符合正态分布的性格特征，且外向性和开放性之间有大约0.4的相关系数
        personality_traits = self.generate_personality_traits()
        self.personality = Personality(*personality_traits)
        
        self.friends = []
        self.alpha = alpha
        self.max_speed = max_speed  # 速度上限

    def generate_personality_traits(self):
        mean = [0, 0, 0, 0, 0]
        cov = [
            [1, 0,   0,   0.4, 0],  # 开放性 (Openness)
            [0, 1,   0,   0,   0],  # 责任心 (Conscientiousness)
            [0, 0,   1,   0,   0],  # 外向性 (Extraversion)
            [0.4, 0,  0,  1,   0],  # 宜人性 (Agreeableness)
            [0, 0,   0,   0,   1]   # 情绪稳定性 (Neuroticism)
        ]
        traits = np.random.multivariate_normal(mean, cov)
        # 将正态分布数据限制在0到1之间
        traits = np.clip(traits, -3, 3)
        traits = (traits - traits.min()) / (traits.max() - traits.min())
        return traits

    def step(self):
        # 基于个性和其他因素移动
        self.move()

        # 与其他代理互动
        self.interact_with_others()

    def move(self):
        # 基于速度方向移动
        new_position = np.array(self.pos) + self.velocity

        # 添加朋友位置的社会重力效应
        social_gravity_force = np.array([0.0, 0.0])
        for friend in self.friends:
            distance_vector = np.array(friend.pos) - np.array(self.pos)
            distance = np.linalg.norm(distance_vector)
            if distance > 0:  # 避免除以零
                social_gravity_force += self.alpha * self.personality.extraversion * distance_vector / (distance ** 2)
        
        # 添加社会重力效应到速度
        self.velocity += social_gravity_force

        # 添加随机变动，影响程度与神经质成正比
        angle_change = random.uniform(-np.pi / 4, np.pi / 4) * self.personality.neuroticism
        speed_change = random.uniform(-0.1, 0.1) * self.personality.neuroticism
        rotation_matrix = np.array([[np.cos(angle_change), -np.sin(angle_change)],
                                    [np.sin(angle_change), np.cos(angle_change)]])
        
        # 应用随机变动
        self.velocity = np.dot(self.velocity, rotation_matrix)
        new_speed = np.linalg.norm(self.velocity) + speed_change
        new_speed = min(new_speed, self.max_speed)  # 限制速度上限

        # 确保速度向量不为零
        if np.linalg.norm(self.velocity) > 0:
            self.velocity = self.velocity / np.linalg.norm(self.velocity) * new_speed
        else:
            self.velocity = np.array([0.0, 0.0])

        # 更新位置
        new_position = np.mod(new_position + self.velocity, [self.model.grid.width, self.model.grid.height])
        new_position = (int(new_position[0]), int(new_position[1]))
        self.model.grid.move_agent(self, new_position)
        self.pos = new_position

    def interact_with_others(self):
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        for agent in cellmates:
            if agent != self:
                self.form_friendship(agent)

    def form_friendship(self, other_agent):
        # 根据性格形成友谊
        if other_agent not in self.friends:
            similarity = self.personality_similarity(other_agent)
            if similarity > 0.5:  # 假设相似度超过0.5会成为朋友
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
