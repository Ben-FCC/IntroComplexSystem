import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
from social_model import SocialModel
import numpy as np

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
    unique, counts = np.unique(values, return_counts=True)
    
    plt.figure()
    plt.plot(unique, counts, 'o-', color='skyblue')  # 点图和折线图
    plt.title('Degree Centrality Distribution')
    plt.xlabel('Degree Centrality')
    plt.ylabel('Frequency')
    plt.show()

model = SocialModel(N=500, width=100, height=100, phi=0.5)

fig, ax = plt.subplots(figsize=(8, 8))
ani = animation.FuncAnimation(fig, update, fargs=(model, ax), frames=200, interval=100, repeat=False)
plt.show()

# 生成最终的网络图
G = nx.Graph()
pos = {}
for agent in model.schedule.agents:
    G.add_node(agent.unique_id)
    pos[agent.unique_id] = (agent.pos[0], agent.pos[1])
    for friend in agent.friends:
        G.add_edge(agent.unique_id, friend.unique_id)

# 绘制度中心性分布图
plot_degree_centrality_distribution(G)
