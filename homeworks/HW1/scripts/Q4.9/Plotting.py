from PGM import population_growth_model
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os 

# get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# if a < 1

AllSimResults = pd.DataFrame()

for a in np.linspace(0.1, 0.9, 5):
    for x0 in np.linspace(0, 2.9, 5):
        
        pgm = population_growth_model(x0=x0, a=a)
        AllSimResults = pd.concat([AllSimResults, pgm.simulate(times=30, plot=False, return_sim=True)], axis=0)
        
sns.relplot(x='UpdateTimes', y='x', col='x0', row='a', kind='line', facet_kws={'sharey': False, 'sharex': True}, data=AllSimResults)
plt.savefig(os.path.join(current_dir, '../../simulation_figures/Q4.9/small_a.png'))

# if a > 1

AllSimResults = pd.DataFrame()

for a in np.linspace(1.5, 3, 5):
    for x0 in np.linspace(0, 2.9, 5):
        
        pgm = population_growth_model(x0=x0, a=a)
        AllSimResults = pd.concat([AllSimResults, pgm.simulate(times=30, plot=False, return_sim=True)], axis=0)
        
sns.relplot(x='UpdateTimes', y='x', col='x0', row='a', kind='line', facet_kws={'sharey': False, 'sharex': True}, data=AllSimResults)
plt.savefig(os.path.join(current_dir, '../../simulation_figures/Q4.9/big_a.png'))