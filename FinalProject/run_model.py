import matplotlib.pyplot as plt
import networkx as nx
from social_model import SocialModel
import numpy as np
from scipy.stats import spearmanr
import seaborn as sns
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
import os
import warnings
import itertools

# 创建存储结果的文件夹
output_dir = "simulation_results"
os.makedirs(output_dir, exist_ok=True)

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
    finished_percent = f'Finished Percent {frame/total_frames*100:.1f}%'
    plt.title(finished_percent)
    local_clustering, global_clustering = calculate_clustering_coefficient(model)
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

def run_simulation(total_frames, phi, sphi, gamma):
    # 忽略所有警告
    warnings.filterwarnings('ignore')
    model = SocialModel(N=100, width=100, height=100, alpha=0.5, max_speed=10.0, break_prob=0.1, phi=phi, sphi=sphi, gamma=gamma)
    time_steps = []
    local_clustering_values = []
    global_clustering_values = []
    agents_data = []

    for frame in range(total_frames):
        model.step()
        time_steps.append(frame)
        local_clustering, global_clustering = calculate_clustering_coefficient(model)
        local_clustering_values.append(local_clustering)
        global_clustering_values.append(global_clustering)

    agents = model.schedule.agents
    for agent in agents:
        personality = agent.personality
        agents_data.append({
            'friends_count': len(agent.friends),
            'openness': personality.openness,
            'conscientiousness': personality.conscientiousness,
            'extraversion': personality.extraversion,
            'agreeableness': personality.agreeableness,
            'neuroticism': personality.neuroticism
        })

    df_clustering = pd.DataFrame({
        'Time Steps': time_steps,
        'Local Clustering Coefficient': local_clustering_values,
        'Global Clustering Coefficient': global_clustering_values
    })

    return df_clustering, agents_data

def main():
    total_frames_list = [5000]
    phi_list = [0.0001, 0.001]
    sphi_list = [0.0001, 0.001]
    gamma_list = [0.0001, 0.001]

    parameter_combinations = list(itertools.product(total_frames_list, phi_list, sphi_list, gamma_list))
    sns.set_theme('talk', 'whitegrid')

    for params in tqdm(parameter_combinations, desc="Running grid search"):
        total_frames, phi, sphi, gamma = params
        print(f"Running simulation with parameters: total_frames={total_frames}, phi={phi}, sphi={sphi}, gamma={gamma}")
        
        # 并行运行模拟
        num_simulations = 10
        results = Parallel(n_jobs=-2)(delayed(run_simulation)(total_frames, phi, sphi, gamma) for _ in range(num_simulations))

        # 合并结果并计算 95% CI
        df_clustering_list = [result[0] for result in results]
        agents_data_list = [result[1] for result in results]

        combined_clustering_df = pd.concat(df_clustering_list)
        combined_agents_data = [item for sublist in agents_data_list for item in sublist]

        clustering_ci_df = combined_clustering_df.groupby('Time Steps').agg(['mean', 'std'])
        clustering_ci_df.columns = ['_'.join(col).strip() for col in clustering_ci_df.columns.values]
        clustering_ci_df['Local_CI_lower'] = clustering_ci_df['Local Clustering Coefficient_mean'] - 1.96 * clustering_ci_df['Local Clustering Coefficient_std']
        clustering_ci_df['Local_CI_upper'] = clustering_ci_df['Local Clustering Coefficient_mean'] + 1.96 * clustering_ci_df['Local Clustering Coefficient_std']
        clustering_ci_df['Global_CI_lower'] = clustering_ci_df['Global Clustering Coefficient_mean'] - 1.96 * clustering_ci_df['Global Clustering Coefficient_std']
        clustering_ci_df['Global_CI_upper'] = clustering_ci_df['Global Clustering Coefficient_mean'] + 1.96 * clustering_ci_df['Global Clustering Coefficient_std']

        # 绘制合并结果
        plt.figure(figsize=(10, 6))
        sns.lineplot(x=clustering_ci_df.index, y='Local Clustering Coefficient_mean', data=clustering_ci_df, label="Local Clustering Coefficient Mean", color='deepskyblue')
        plt.fill_between(clustering_ci_df.index, clustering_ci_df['Local_CI_lower'], clustering_ci_df['Local_CI_upper'], color='deepskyblue', alpha=0.3)
        sns.lineplot(x=clustering_ci_df.index, y='Global Clustering Coefficient_mean', data=clustering_ci_df, label="Global Clustering Coefficient Mean", color='red')
        plt.fill_between(clustering_ci_df.index, clustering_ci_df['Global_CI_lower'], clustering_ci_df['Global_CI_upper'], color='red', alpha=0.3)
        plt.title(f'Clustering Coefficient vs Time Steps with 95% CI\n(total_frames={total_frames}, phi={phi}, sphi={sphi}, gamma={gamma})')
        plt.xlabel('Time Steps')
        plt.ylabel('Clustering Coefficient')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'clustering_coefficient_vs_time_steps_with_ci_frames{total_frames}_phi{phi}_sphi{sphi}_gamma{gamma}.png'))
        plt.close()

        # 生成 personality vs. friendship 数量的 lmplot
        df_agents = pd.DataFrame(combined_agents_data)
        df_ranked = df_agents.rank(pct=True)
        personality_traits = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
        colors = ['blue', 'orange', 'red', 'salmon', 'purple']
        for trait, color in zip(personality_traits, colors):
            spearman_corr, _ = spearmanr(df_agents['friends_count'], df_agents[trait])
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=trait, y='friends_count', data=df_ranked, alpha=0.5, s=20, color=color)
            sns.regplot(x=trait, y='friends_count', data=df_ranked, scatter=False, color=color)
            plt.title(f'Friends Count vs. {trait.capitalize()} (Spearman Corr: {spearman_corr:.2f})\n(total_frames={total_frames}, phi={phi}, sphi={sphi}, gamma={gamma})')
            plt.xlabel(trait.capitalize())
            plt.ylabel('Friends Count')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'lmplot_{trait}_frames{total_frames}_phi{phi}_sphi{sphi}_gamma{gamma}.png'))
            plt.close()

if __name__ == "__main__":
    main()
