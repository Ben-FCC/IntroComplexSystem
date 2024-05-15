import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define the Game of Life rules
def game_of_life_step(grid):
    neighbors = sum(np.roll(np.roll(grid, i, 0), j, 1)
                    for i in (-1, 0, 1) for j in (-1, 0, 1)
                    if (i != 0 or j != 0))
    return (neighbors == 3) | (grid & (neighbors == 2))

# Function to run the simulation for a given number of iterations
def run_simulation(p0, size, iterations):
    grid = np.random.choice([0, 1], size=(size, size), p=[1-p0, p0])
    densities = []

    for _ in range(iterations):
        grid = game_of_life_step(grid)
        densities.append(np.mean(grid))
    
    return densities

# Parameters
size = 100
iterations = 100
monte_carlo_runs = 100

for p0 in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    # Run Monte Carlo simulation
    all_densities = []

    for _ in range(monte_carlo_runs):
        densities = run_simulation(p0, size, iterations)
        all_densities.append(densities)

    # Convert to numpy array for easier manipulation
    all_densities = np.array(all_densities)

    # Calculate mean and 95% confidence interval
    mean_densities = np.mean(all_densities, axis=0)
    std_densities = np.std(all_densities, axis=0)
    ci_lower = mean_densities - 1.96 * std_densities / np.sqrt(monte_carlo_runs)
    ci_upper = mean_densities + 1.96 * std_densities / np.sqrt(monte_carlo_runs)

    # Mean Field Approximation
    def MeanFieldApproximation_GL2D(p):
        return p * (28 * p**2 * (1 - p)**6 + 56 * p**3 * (1 - p)**5) + (1 - p) * (56 * p**3 * (1 - p)**5)

    # Iterate MFA for the same number of steps as the simulation
    mfa_values = [p0]  # Initial density
    for _ in range(iterations - 1):
        mfa_values.append(MeanFieldApproximation_GL2D(mfa_values[-1]))

    # Plotting
    sns.set_context('talk')

    plt.figure(figsize=(12, 8))
    plt.plot(mean_densities, label='Simulation Mean')
    plt.fill_between(range(iterations), ci_lower, ci_upper, color='b', alpha=0.2, label='95% CI')
    plt.plot(range(iterations), mfa_values, label='Mean Field Approximation', linestyle='--')
    plt.xlabel('Iterations')
    plt.ylabel('Average Density')
    plt.title('Comparison of Mean Field Approximation and Simulation with 95% CI')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'../../simulation_figures/Q12.7/MeanFieldApproximation_vs_Simulation_p0={p0}.png')
