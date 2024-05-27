from social_model import SocialModel
from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 定義每個代理的呈現方式
def agent_portrayal(agent):
    portrayal = {
        "Shape": "circle",
        "Filled": "true",
        "r": 0.5,
        "Color": "red",
        "Layer": 0,
        "text": f"{len(agent.friends)}",
        "text_color": "white"
    }
    return portrayal

# 視覺化參數設置
grid = CanvasGrid(agent_portrayal, 10, 10, 500, 500)

server = ModularServer(
    SocialModel,
    [grid],
    "Social Model",
    {"N": 10, "width": 10, "height": 10, "alpha": 0.1}
)

# 動態可視化設置
fig, ax = plt.subplots(figsize=(10, 10))

def update(num, model, G, ax):
    ax.clear()
    model.step()
    
    # 清除之前的邊
    G.clear_edges()
    
    # 添加新的邊
    for agent in model.schedule.agents:
        for friend in agent.friends:
            G.add_edge(agent.unique_id, friend.unique_id)
    
    # 繪製網絡圖
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=500, node_color="skyblue", font_size=10, font_color="black", font_weight="bold", edge_color="gray", ax=ax)
    ax.set_title(f'Step {num}')

# 初始化模型和圖
model = SocialModel(10, 10, 10, alpha=0.1)
G = nx.Graph()
for agent_id in range(model.num_agents):
    G.add_node(agent_id)

# 創建動畫
ani = animation.FuncAnimation(fig, update, frames=100, fargs=(model, G, ax), interval=200, repeat=False)

plt.show()
