from social_model import SocialModel
from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer
import matplotlib.pyplot as plt

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
    {"N": 10, "width": 10, "height": 10}
)

# 運行伺服器
server.port = 8521  # The default
server.launch()

# 進行數據收集和模擬運行
model = SocialModel(10, 10, 10)
for i in range(100):
    model.step()

# 收集數據
data = model.datacollector.get_agent_vars_dataframe()
print(data)

# 視覺化代理位置和友誼
positions = data.xs(99, level="Step")["Position"]
friends = data.xs(99, level="Step")["Friends"]

plt.figure(figsize=(10, 10))
plt.xlim(0, 10)
plt.ylim(0, 10)

for i, pos in enumerate(positions):
    plt.scatter(pos[0], pos[1], s=100)
    plt.text(pos[0], pos[1], f"{friends[i]}", fontsize=12, ha='right')

plt.title('Agent Positions and Friendships')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.show()
