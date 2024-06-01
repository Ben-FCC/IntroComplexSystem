import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
from social_model import SocialModel
import numpy as np
from scipy.stats import spearmanr

def draw_network(G, pos, ax, grid_width, grid_height):
    ax.clear()
    node_color = 'skyblue'
    node_size = 300
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_size, node_color=node_color, alpha=0.9, edgecolors='black')
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.5, edge_color='gray')
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=10)
    ax.set_xlim(-1, grid_width + 1)
    ax.set_ylim(-1, grid_height + 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Social Network of Agents')

def update(frame, model, ax):
    model.step()
    G = nx.Graph()
    pos = {}
    for agent in model.schedule.agents:
        G.add_node(agent.unique_id)
        pos[agent.unique_id] = (agent.pos[0], agent.pos[1])
        for friend in agent.friends:
            G.add_edge(agent.unique_id, friend.unique_id)
    draw_network(G, pos, ax, model.grid.width, model.grid.height)
    return G

def plot_degree_centrality_distribution(G):
    degree_centrality = nx.degree_centrality(G)
    values = list(degree_centrality.values())
    
    plt.figure()
    plt.hist(values, bins=20, color='skyblue', edgecolor='black')  # 直方图
    plt.title('Degree Centrality Distribution')
    plt.xlabel('Degree Centrality')
    plt.ylabel('Frequency')
    plt.show()

def plot_personality_network(G, agents, personality_trait, title):
    pos = nx.spring_layout(G, seed=42, k=0.5)  # 使用 spring_layout，并增加 k 值以确保节点分布更加均匀
    node_color = [getattr(agent.personality, personality_trait) for agent in agents]  # 节点颜色由 personality trait 决定
    node_size = [300 * np.log1p(len(agent.friends)) for agent in agents]  # 节点大小与 friends 数量的 log 成正比

    # 计算 Spearman 相关系数
    friends_counts = [len(agent.friends) for agent in agents]
    personality_values = [getattr(agent.personality, personality_trait) for agent in agents]
    spearman_corr, _ = spearmanr(friends_counts, personality_values)

    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_size, node_color=node_color, cmap=plt.cm.coolwarm, alpha=0.9, edgecolors='black')
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.5, edge_color='gray')
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'{title} Network (Spearman Corr: {spearman_corr:.2f})')
    plt.colorbar(nx.draw_networkx_nodes(G, pos, node_color=node_color, cmap=plt.cm.coolwarm, alpha=0.9), ax=ax, label='Personality Value')
    plt.show()

model = SocialModel(N=100, width=100, height=100, alpha=0.5, max_speed=3.0, break_prob=0.05, phi=0.1)

fig, ax = plt.subplots(figsize=(8, 8))
ani = animation.FuncAnimation(fig, update, fargs=(model, ax), frames=200, interval=100, repeat=False)
plt.show()

# 生成最终的网络图
G = nx.Graph()
agents = model.schedule.agents
pos = {}
for agent in agents:
    G.add_node(agent.unique_id)
    pos[agent.unique_id] = (agent.pos[0], agent.pos[1])
    for friend in agent.friends:
        G.add_edge(agent.unique_id, friend.unique_id)

# 绘制度中心性分布图
plot_degree_centrality_distribution(G)

# 根据不同的人格特质生成网络图
personality_traits = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
for trait in personality_traits:
    plot_personality_network(G, agents, trait, f'{trait.capitalize()} Network')

# 计算聚类系数
local_clustering = nx.average_clustering(G)
global_clustering = nx.transitivity(G)

print(f'Local Clustering Coefficient: {local_clustering}')
print(f'Global Clustering Coefficient: {global_clustering}')
