from Fibonacci import Fibonacci_model
import matplotlib.pyplot as plt
import os
import seaborn as sns

# get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

fm = Fibonacci_model()
SimResults = fm.simulate(times=30, return_sim=True)

# pltting the results

## plot the x over time
sns.lineplot(x='UpdateTimes', y='x', data=SimResults)
plt.title(f'UpdateTimes: {len(fm.UpdateTimes)-1}')
plt.tight_layout()
plt.savefig(os.path.join(current_dir, '../../simulation_figures/Q4.8/x_by_time.png'))
plt.clf()

## plot phase space
sns.lineplot(y='x', x='y', data=SimResults)
plt.title(f'Phase Space, UpdateTimes: {len(fm.UpdateTimes)-1}')
plt.tight_layout()
plt.savefig(os.path.join(current_dir, '../../simulation_figures/Q4.8/Phase_Space.png'))