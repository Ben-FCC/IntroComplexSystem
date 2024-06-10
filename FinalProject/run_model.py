import matplotlib.pyplot as plt
import networkx as nx
from social_model import SocialModel
import numpy as np
import seaborn as sns
import os
from joblib import Parallel, delayed
from tqdm import tqdm
import warnings

# 创建存储结果的文件夹
output_dir = "simulation_results"
os.makedirs(output_dir, exist_ok=True)

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

# Define the range of parameters using log scale
phi_values = [0.00001]
sphi_values = np.logspace(-5, 0, 10)
gamma_values = np.logspace(-5, 0, 10)
num_simulations = 3  # Number of Monte Carlo simulations

# Function to run the model and collect clustering coefficients
def run_simulation(phi, sphi, gamma):
    warnings.filterwarnings("ignore")
    local_clustering_list = []
    global_clustering_list = []

    for _ in range(num_simulations):
        model = SocialModel(100, 100, 100, phi=phi, sphi=sphi, gamma=gamma)
        for _ in range(3000):
            model.step()
        local_clustering, global_clustering = calculate_clustering_coefficient(model)
        local_clustering_list.append(local_clustering)
        global_clustering_list.append(global_clustering)

    avg_local_clustering = np.mean(local_clustering_list)
    avg_global_clustering = np.mean(global_clustering_list)

    return avg_local_clustering, avg_global_clustering

# Wrapper to handle the parallel processing of simulations
def parallel_simulation(phi, sphi_values, gamma_values):
    results = Parallel(n_jobs=-3)(
        delayed(run_simulation)(phi, sphi, gamma) for sphi, gamma in tqdm(
            [(sphi, gamma) for sphi in sphi_values for gamma in gamma_values],
            desc=f"Simulations for phi={phi}", total=len(sphi_values)*len(gamma_values)))
    local_clustering_results = np.array([result[0] for result in results]).reshape(len(gamma_values), len(sphi_values))
    global_clustering_results = np.array([result[1] for result in results]).reshape(len(gamma_values), len(sphi_values))
    return local_clustering_results, global_clustering_results

# Iterate over fixed phi values
for phi in phi_values:
    local_clustering_results, global_clustering_results = parallel_simulation(phi, sphi_values, gamma_values)

    # Create and save heatmaps
    plt.figure(figsize=(10, 6))
    sns.heatmap(global_clustering_results, cmap='inferno', cbar_kws={'label': 'Global Clustering Coefficient'})
    plt.title(f"Global Clustering Coefficient for phi = {phi}")
    plt.xlabel("sphi")
    plt.ylabel("gamma")
    plt.xticks(ticks=np.arange(len(sphi_values)), labels=[f'{val:.1e}' for val in sphi_values], rotation=45)
    plt.yticks(ticks=np.arange(len(gamma_values)), labels=[f'{val:.1e}' for val in gamma_values], rotation=0)
    plt.savefig(os.path.join(output_dir, f"global_clustering_phi_{phi}.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.heatmap(local_clustering_results, cmap='inferno', cbar_kws={'label': 'Local Clustering Coefficient'})
    plt.title(f"Local Clustering Coefficient for phi = {phi}")
    plt.xlabel("sphi")
    plt.ylabel("gamma")
    plt.xticks(ticks=np.arange(len(sphi_values)), labels=[f'{val:.1e}' for val in sphi_values], rotation=45)
    plt.yticks(ticks=np.arange(len(gamma_values)), labels=[f'{val:.1e}' for val in gamma_values], rotation=0)
    plt.savefig(os.path.join(output_dir, f"local_clustering_phi_{phi}.png"))
    plt.close()
