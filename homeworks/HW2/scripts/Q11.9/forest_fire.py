import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from joblib import Parallel, delayed
import pandas as pd

# Initialize forest with given density
def initialize_forest(size, p):
    forest = np.random.choice([0, 1], size=(size, size), p=[1-p, p])
    return forest

# Step function to update the forest
def step(forest):
    new_forest = forest.copy()
    size = forest.shape[0]
    
    for i in range(size):
        for j in range(size):
            if forest[i, j] == 2:
                new_forest[i, j] = 3  # burning tree becomes charred
                neighbors = [(i-1, j), (i+1, j), (i, j-1), (i, j+1),
                             (i-1, j-1), (i-1, j+1), (i+1, j-1), (i+1, j+1)]  # Moore
                for ni, nj in neighbors:
                    if 0 <= ni < size and 0 <= nj < size and forest[ni, nj] == 1:
                        new_forest[ni, nj] = 2  # tree catches fire
    return new_forest

# Simulate fire spread until it stops
def simulate_fire(size, p, steps=1000):
    forest = initialize_forest(size, p)
    # Start the fire at the center
    center = size // 2
    forest[center, center] = 2
    
    for step_count in range(steps):
        new_forest = step(forest)
        if np.array_equal(forest, new_forest):
            break
        forest = new_forest
    
    burned_area = np.sum(forest == 3)
    return burned_area, step_count

# Monte Carlo simulation with parallel processing
def monte_carlo_simulation(size, p_values, steps=1000, runs=100, n_jobs=-3):
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

# Convert results to DataFrames for easier plotting with seaborn
df_burned_area = pd.DataFrame(results_burned_area, columns=['p', 'burned_area'])
df_stop_time = pd.DataFrame(results_stop_time, columns=['p', 'stop_time'])

sns.set_context('talk')

# Plot the results for burned area
plt.figure(figsize=(10, 6))
sns.lineplot(x='p', y='burned_area', data=df_burned_area, ci=95)
plt.xlabel('Probability of tree presence (p)')
plt.ylabel('Average burned area')
plt.title('Average burned area as a function of p with 95% CI')
plt.tight_layout()
plt.savefig('../../simulation_figures/Q11.9/burned area.png')

# Plot the results for stop time
plt.figure(figsize=(10, 6))
sns.lineplot(x='p', y='stop_time', data=df_stop_time, ci=95)
plt.xlabel('Probability of tree presence (p)')
plt.ylabel('Time until fire stops')
plt.title('Time until fire stops as a function of p with 95% CI')
plt.tight_layout()
plt.savefig('../../simulation_figures/Q11.9/time_stop.png')

# Compare the final results for different values of p
# Print summary statistics for both burned area and stop time
burned_area_summary = df_burned_area.groupby('p')['burned_area'].describe()
stop_time_summary = df_stop_time.groupby('p')['stop_time'].describe()

print("Summary statistics for burned area:")
print(burned_area_summary)

print("\nSummary statistics for stop time:")
print(stop_time_summary)
