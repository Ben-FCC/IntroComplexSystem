import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
from social_model import SocialModel
import numpy as np
from scipy.stats import spearmanr
import seaborn as sns
import pandas as pd

def draw_network(G, pos, ax, grid_width, grid_height, finished_percent):
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
    ax.set_title('Social Network of Agents\n' + finished_percent, fontsize=16)

def update(frame, model, total_frames, ax, plot=False):
    model.step()
    time_steps.append(frame)
    finished_percent = f'Finished Percent {frame/total_frames*100:.1f}%'
    plt.title(finished_percent)
    local_clustering, global_clustering = calculate_clustering_coefficient(model)
    local_clustering_values.append(local_clustering)
    global_clustering_values.append(global_clustering)
    if plot:
        G = nx.Graph()
        pos = {}
        for agent in model.schedule.agents:
            G.add_node(agent.unique_id)
            pos[agent.unique_id] = (agent.pos[0], agent.pos[1])
            for friend in agent.friends:
                G.add_edge(agent.unique_id, friend.unique_id)
        draw_network(G, pos, ax, model.grid.width, model.grid.height, finished_percent)
        return G, local_clustering, global_clustering

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

def calculate_clustering_coefficient(model):
    G = nx.Graph()
    agents = model.schedule.agents
    for agent in agents:
        G.add_node(agent.unique_id)
        for friend in agent.friends:
            G.add_edge(agent.unique_id, friend.unique_id)

    local_clustering = nx.average_clustering(G)
    global_clustering = nx.transitivity(G)
    return local_clustering, global_clustering

model = SocialModel(N=100, width=100, height=100, alpha=0.5, max_speed=10.0, break_prob=0.1, phi=0.00005, sphi=0.00005, gamma=0.000)

sns.set_theme(context='talk', style="whitegrid")
fig, ax = plt.subplots(figsize=(8, 8))
total_frames = 10000
time_steps = []
local_clustering_values = []
global_clustering_values = []
ani = animation.FuncAnimation(fig, update, fargs=(model, total_frames, ax, False), frames=total_frames, interval=100, repeat=False)
plt.show()

# Plot clustering coefficient against time steps
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
df = pd.DataFrame({'Time Steps': time_steps, 'Local Clustering Coefficient': local_clustering_values, 'Global Clustering Coefficient': global_clustering_values})
sns.lineplot(x=time_steps, y=local_clustering_values, label="Local Clustering Coefficient", color='deepskyblue', data=df)
sns.lineplot(x=time_steps, y=global_clustering_values, label="Global Clustering Coefficient", color='red', data=df)
plt.title('Clustering Coefficient vs Time Steps')
plt.xlabel('Time Steps')
plt.ylabel('Clustering Coefficient')
plt.legend()
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
