import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from joblib import Parallel, delayed
import pandas as pd

def initialize_forest(size, p):
    """Initialize the forest with trees distributed with probability p."""
    return np.random.choice([0, 1], size=(size, size), p=[1-p, p])

def step(forest):
    """Perform a single step of the fire spreading simulation."""
    new_forest = forest.copy()
    size = forest.shape[0]
    
    for i in range(size):
        for j in range(size):
            if forest[i, j] == 2:
                new_forest[i, j] = 3  # Burning tree becomes charred
                neighbors = [(i-1, j), (i+1, j), (i, j-1), (i, j+1),
                             (i-1, j-1), (i-1, j+1), (i+1, j-1), (i+1, j+1)]  # Moore neighborhood
                for ni, nj in neighbors:
                    if 0 <= ni < size and 0 <= nj < size and forest[ni, nj] == 1:
                        new_forest[ni, nj] = 2  # Tree catches fire
    return new_forest

def simulate_fire(size, p, steps=1000):
    """Simulate the fire spread and return the burned area and time until it stops."""
    forest = initialize_forest(size, p)
    center = size // 2
    forest[center, center] = 2  # Start fire at the center
    
    for t in range(steps):
        new_forest = step(forest)
        if np.array_equal(forest, new_forest):
            break
        forest = new_forest
    
    burned_area = np.sum(forest == 3)
    return burned_area, t

def monte_carlo_simulation(size, p_values, steps=1000, runs=100, n_jobs=-3):
    """Perform Monte Carlo simulation over a range of p values with parallel processing."""
    results_burned_area = []
    results_stop_time = []
    
    for p in tqdm(p_values, desc="Simulating for different p values"):
        simulations = Parallel(n_jobs=n_jobs)(delayed(simulate_fire)(size, p, steps) for _ in range(runs))
        burned_areas, stop_times = zip(*simulations)
        for burned_area, stop_time in zip(burned_areas, stop_times):
            results_burned_area.append((p, burned_area))
            results_stop_time.append((p, stop_time))
    
    return results_burned_area, results_stop_time

# Parameters
size = 100
p_values = np.linspace(0.1, 0.9, 9)
steps = 1000
runs = 100

# Run the Monte Carlo simulation
results_burned_area, results_stop_time = monte_carlo_simulation(size, p_values, steps, runs)

# Plotting one trial of CA spatial simulations per p for comparison
def plot_one_trial_per_p(size, p_values):
    cmap = plt.cm.get_cmap('gray', 4)  # Create a colormap with 4 colors
    colors = {0: 'lightgray', 1: 'green', 2: 'red', 3: 'black'}
    labels = {0: 'Empty', 1: 'Tree', 2: 'Burning', 3: 'Charred'}

    plt.figure(figsize=(15, 15))
    for i, p in enumerate(p_values):
        forest = initialize_forest(size, p)
        center = size // 2
        forest[center, center] = 2  # Start fire at the center
        for _ in range(steps):
            forest = step(forest)
            if np.sum(forest == 2) == 0:
                break
        
        plt.subplot(3, 3, i+1)
        img = np.zeros((size, size, 3))
        for state, color in colors.items():
            img[np.where(forest == state)] = plt.cm.colors.to_rgb(color)
        plt.imshow(img)
        plt.title(f'p = {p:.2f}')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('../../simulation_figures/Q11.9/one_trial_per_p.png')

# Plot one trial per p
plot_one_trial_per_p(size, p_values)
