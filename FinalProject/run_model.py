import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
from social_model import SocialModel

def draw_network(G, pos, ax, width, height):
    ax.clear()
    # 设置节点的颜色和大小
    node_color = 'skyblue'
    node_size = 300

    # 绘制节点和边
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_size, node_color=node_color, alpha=0.9, edgecolors='black')
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.5, edge_color='gray')
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=10)

    # 设置轴的范围
    ax.set_xlim(-1, width + 1)
    ax.set_ylim(-1, height + 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Social Network of Agents')

def update(frame, model, ax):
    model.step()
    G = nx.Graph()
    pos = {}
    for agent in model.schedule.agents:
        G.add_node(agent.unique_id)
        pos[agent.unique_id] = (agent.pos[0], agent.pos[1])  # 使用代理的实际位置作为节点位置
        for friend in agent.friends:
            G.add_edge(agent.unique_id, friend.unique_id)
    draw_network(G, pos, ax, model.grid.width, model.grid.height)

# 初始化模型和图形
model = SocialModel(N=100, width=100, height=100, alpha=0.5, max_speed=3.0)
fig, ax = plt.subplots(figsize=(10, 10))

# 创建动画
ani = animation.FuncAnimation(fig, update, frames=100, fargs=(model, ax), interval=200, repeat=False)
plt.show()
