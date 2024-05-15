import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def MeanFieldApproximation_GL2D(p):
    return p * (28 * p**2 * (1 - p)**6 + 56 * p**3 * (1 - p)**5) + (1 - p) * (56 * p**3 * (1 - p)**5)

# Parameters for cobweb plot
p_min, p_max = 0, 1
iterations = 100  # Number of iterations for cobweb plot

# Function for cobweb plot
def cobweb_plot(func, p0, iterations, p_min, p_max):
    p = np.linspace(p_min, p_max, 1000)
    plt.plot(p, p, label='$p_{t} = p_{t-1}$')
    plt.plot(p, func(p), label='Mean Field Approximation')

    x, y = p0, func(p0)
    for _ in range(iterations):
        plt.plot([x, x], [x, y], color='k', lw=1)
        plt.plot([x, y], [y, y], color='k', lw=1)
        x, y = y, func(y)

    plt.xlim(p_min, p_max)
    plt.ylim(p_min, p_max)
    plt.xlabel('p')
    plt.ylabel('Mean Field Approximation')
    plt.title(f'Cobweb Plot for \nMean Field Approximation for 2D GL model\nwith initial condition p0 = {initial_p}')
    #plt.legend()
    plt.grid(True)
    plt.tight_layout()

# Plot the cobweb diagram
sns.set_context('talk')
for initial_p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    cobweb_plot(MeanFieldApproximation_GL2D, initial_p, iterations, p_min, p_max)
    plt.savefig(f'../../simulation_figures/Q12.7/MeanFieldApproximation_GL2D_p0={initial_p}.png')
    plt.clf()
