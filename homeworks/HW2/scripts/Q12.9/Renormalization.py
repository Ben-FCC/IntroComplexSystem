import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Renormalization Approximation for forest fire wiht Neumann neighborhood
def Renormalization(p):
    return -p**4 + 2*p**2

# Parameters for cobweb plot
p_min, p_max = 0, 1
iterations = 100  # Number of iterations for cobweb plot

# Function for cobweb plot
def cobweb_plot(func, p0, iterations, p_min, p_max):
    p = np.linspace(p_min, p_max, 1000)
    plt.plot(p, p, label='$p_{t} = p_{t-1}$')
    plt.plot(p, func(p), label='Renormalization Approximation')

    x, y = p0, func(p0)
    for _ in range(iterations):
        plt.plot([x, x], [x, y], color='k', lw=1)
        plt.plot([x, y], [y, y], color='k', lw=1)
        x, y = y, func(y)

    plt.xlim(p_min, p_max)
    plt.ylim(p_min, p_max)
    plt.xlabel('p')
    plt.ylabel('Renormalization Approximation')
    plt.title(f'Cobweb Plot for \nRenormalization Approximation for forest fire\nwith initial condition p0 = {initial_p}')
    #plt.legend()
    plt.grid(True)
    plt.tight_layout()

# Plot the cobweb diagram
sns.set_context('talk')
for initial_p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    cobweb_plot(Renormalization, initial_p, iterations, p_min, p_max)
    plt.savefig(f'../../simulation_figures/Q12.9/Renormalization_Approximation_p0={initial_p}.png')
    plt.clf()
