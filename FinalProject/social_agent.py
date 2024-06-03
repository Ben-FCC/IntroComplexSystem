from mesa import Agent
import numpy as np
import random
from personality import Personality
import networkx as nx

class SocialAgent(Agent):
    def __init__(self, unique_id, model, alpha=0.1, max_speed=3.0, break_prob=0.05, phi=0.01, sphi=0.01):
        super().__init__(unique_id, model)
        self.pos = (random.randrange(model.grid.width), random.randrange(model.grid.height))

        self.speed = np.clip(np.abs(np.random.normal(1, 1)), 0, max_speed)
        self.velocity = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])
        self.velocity = self.velocity / np.linalg.norm(self.velocity) * self.speed

        personality_traits = self.generate_personality_traits()
        self.personality = Personality(*personality_traits)

        self.friends = []
        self.alpha = alpha
        self.max_speed = max_speed
        self.break_prob = break_prob
        self.phi = phi
        self.sphi = sphi

    def generate_personality_traits(self):
        mean = [0, 0, 0, 0, 0]
        cov = [
            [1.0, 0.3, 0.4, 0.2, -0.1],
            [0.3, 1.0, 0.5, 0.4, -0.2],
            [0.4, 0.5, 1.0, 0.6, -0.3],
            [0.2, 0.4, 0.6, 1.0, -0.4],
            [-0.1, -0.2, -0.3, -0.4, 1.0]
        ]
        traits = np.random.multivariate_normal(mean, cov)
        traits = np.clip(traits, -3, 3)
        traits = (traits - traits.min()) / (traits.max() - traits.min())
        return traits

    def step(self):
        self.move()
        self.interact_with_others()
        self.break_friendships()

    def move(self):
        new_position = np.array(self.pos) + self.velocity
        social_gravity_force = np.array([0.0, 0.0])
        for friend in self.friends:
            distance_vector = np.array(friend.pos) - np.array(self.pos)
            distance = np.linalg.norm(distance_vector)
            if distance > 0:
                force = self.alpha * distance_vector / distance
                social_gravity_force += force
        self.velocity += social_gravity_force
        self.velocity = np.clip(self.velocity, -self.max_speed, self.max_speed)
        speed_change = np.random.normal(0, 0.1)
        angle_change = np.random.normal(0, np.pi / 4)
        rotation_matrix = np.array([
            [np.cos(angle_change), -np.sin(angle_change)],
            [np.sin(angle_change), np.cos(angle_change)]
        ])
        self.velocity = np.dot(self.velocity, rotation_matrix)
        new_speed = np.linalg.norm(self.velocity) + speed_change
        new_speed = min(new_speed, self.max_speed)
        if np.linalg.norm(self.velocity) > 0:
            self.velocity = self.velocity / np.linalg.norm(self.velocity) * new_speed
        else:
            self.velocity = np.array([0.0, 0.0])
        new_position = np.mod(new_position + self.velocity, [self.model.grid.width, self.model.grid.height])
        new_position = (int(new_position[0]), int(new_position[1]))
        self.model.grid.move_agent(self, new_position)
        self.pos = new_position

    def interact_with_others(self):
        all_agents = self.model.schedule.agents
        for agent in all_agents:
            if agent != self:
                self.form_friendship(agent)

    def form_friendship(self, other_agent):
        distance_vector = np.array(other_agent.pos) - np.array(self.pos)
        distance = np.linalg.norm(distance_vector)
        if distance > 0:
            #social_distance = self.calculate_social_distance(other_agent)
            personality_distance = self.calculate_personality_distance(other_agent)
            #probability = self.phi * (1 / (distance ** 2)) + self.sphi * (1 / ((social_distance * personality_distance) ** 2))
            probability = self.phi * (1 / (distance ** 2)) + self.sphi * (1 / ((personality_distance) ** 2))
            #probability = 1 / (1 + np.exp(-self.phi * (1 / ((distance ** 2 ) * social_distance * personality_distance))))
            if random.random() < probability:
                if other_agent not in self.friends:
                    self.friends.append(other_agent)
                if self not in other_agent.friends:
                    other_agent.friends.append(self)

    def break_friendships(self):
        epsilon = 1e-6  # 避免除以零
        for friend in self.friends[:]:  # 使用副本来避免在遍历时修改列表
            # 断裂概率与宜人性反比，与各自的神经质成正比
            break_probability = (self.break_prob * self.personality.neuroticism * friend.personality.neuroticism) / (self.personality.agreeableness * friend.personality.agreeableness + epsilon)
            if random.random() < break_probability:
                self.friends.remove(friend)
                if self in friend.friends:
                    friend.friends.remove(self)

    def calculate_social_distance(self, other_agent):
        graph = nx.Graph()
        for agent in self.model.schedule.agents:
            for friend in agent.friends:
                graph.add_edge(agent.unique_id, friend.unique_id)
        if self.unique_id in graph and other_agent.unique_id in graph:
            try:
                distance = nx.shortest_path_length(graph, self.unique_id, other_agent.unique_id)
                return min(distance, 10)  # 若最短路徑大於 10，則返回 10
            except nx.NetworkXNoPath:
                return 10  # 設定最大 path 為 10
        else:
            return 10  # 設定最大 path 為 10

    def calculate_personality_distance(self, other_agent):
        distance = np.linalg.norm([
            self.personality.openness - other_agent.personality.openness,
            self.personality.conscientiousness - other_agent.personality.conscientiousness,
            self.personality.extraversion - other_agent.personality.extraversion,
            self.personality.agreeableness - other_agent.personality.agreeableness,
            self.personality.neuroticism - other_agent.personality.neuroticism
        ]) / np.sqrt(5)
        return distance
