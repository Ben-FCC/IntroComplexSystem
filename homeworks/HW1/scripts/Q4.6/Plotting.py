from SLM import simple_linear_model
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# if a < -0.5

AllSimResults = pd.DataFrame()

for a in np.linspace(-2.5, -0.5, 5):
    for b in np.linspace(-2, 2, 5):
        
        slm = simple_linear_model(x0=1, a=a, b=b)
        AllSimResults = pd.concat([AllSimResults, slm.simulate(times=30, 
        plot=False, return_sim=True)], axis=0)
        
sns.relplot(x='UpdateTimes', y='x', col='b', row='a', kind='line', facet_kws={'sharey': False, 'sharex': True}, data=AllSimResults)

plt.savefig(os.path.join(current_dir, '../../simulation_figures/Q4.6/negative_a.png'))

# if a > 0 

AllSimResults = pd.DataFrame()

for a in np.linspace(0.5, 2.5, 5):
    for b in np.linspace(-2, 2, 5):
        
        slm = simple_linear_model(x0=1, a=a, b=b)
        AllSimResults = pd.concat([AllSimResults, slm.simulate(times=30, 
                        plot=False, return_sim=True)], axis=0)
        
sns.relplot(x='UpdateTimes', y='x', col='b', row='a', kind='line', facet_kws={'sharey': False, 'sharex': True}, data=AllSimResults)

plt.savefig(os.path.join(current_dir, '../../simulation_figures/Q4.6/positive_a.png'))