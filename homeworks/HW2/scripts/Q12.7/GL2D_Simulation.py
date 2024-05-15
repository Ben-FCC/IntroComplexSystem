import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define the Game of Life rules
def game_of_life_step(grid):
    neighbors = sum(np.roll(np.roll(grid, i, 0), j, 1)
                    for i in (-1, 0, 1) for j in (-1, 0, 1)
                    if (i != 0 or j != 0))
    return (neighbors == 3) | (grid & (neighbors == 2))

# Initialize the grid
size = 100
grid = np.random.choice([0, 1], size=(size, size), p=[0.5, 0.5])

# Run the simulation and track the average density
iterations = 100
densities = []

for _ in range(iterations):
    grid = game_of_life_step(grid)
    densities.append(np.mean(grid))

# Mean Field Approximation
def MeanFieldApproximation_GL2D(p):
    return p * (28 * p**2 * (1 - p)**6 + 56 * p**3 * (1 - p)**5) + (1 - p) * (56 * p**3 * (1 - p)**5)

# Plotting
sns.set_context('talk')
p = np.linspace(0, 1, 100)
mfa = [MeanFieldApproximation_GL2D(pi) for pi in p]

plt.plot(densities, label='Simulation', color='deepskyblue')
plt.plot(p, mfa, label='Mean Field Approximation', color='red')
plt.xlabel('Iterations')
plt.ylabel('Average Density')
plt.title('Mean Field Approximation vs. Simulation')
plt.legend()
plt.tight_layout()
plt.savefig('../../simulation_figures/Q12.7/MeanFieldApproximation_vs_Simulation.png')
