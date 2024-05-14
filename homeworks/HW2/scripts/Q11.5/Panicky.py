import matplotlib.pyplot as plt
import numpy as np
from pylab import zeros, imshow, cla, argmax, random, savefig, pause
from tqdm import tqdm, trange
import seaborn as sns
import pandas as pd
from joblib import Parallel, delayed

n = 100 # size of space: n x n

class Panicky_Agent(object):

    def __init__(self, p, n_states):
        self.config = zeros([n, n])
        self.nextconfig = zeros([n, n])
        for x in range(n):
            for y in range(n):
                for s in range(n_states-1):
                    if self.config[x, y] == 0:
                        self.config[x, y] = s+1 if random() < p else 0
                    else:
                        continue
                    self.nextconfig = zeros([n, n])
        
    def observe(self):
        # getting the original colormap using cm.get_cmap() function 
        orig_map=plt.cm.get_cmap('Pastel1') 
        
        # reversing the original colormap using reversed() function 
        reversed_map = orig_map.reversed() 
        cla()
        imshow(self.config, vmin = 0, vmax = n_states-1, cmap = reversed_map)
        
    def stable(self):
        if np.sum(self.nextconfig != self.config):
            return False
        else:
            return True

    def density_of_panicky(self):
        return np.sum(self.config == 1) / n**2
        
    def update(self):
        for x in range(n):
            for y in range(n):
                count = np.zeros(n_states)
                for s in range(n_states):
                    for dx in [-r, 0, r]:
                        for dy in [-r, 0, r]:
                            if self.config[(x + dx) % n, (y + dy) % n] == s:
                                count[s] += 1
                self.nextconfig[x, y] = 1 if count[1] >= 4 else 0
        self.config, self.nextconfig = self.nextconfig, self.config


r = 1
n_states = 2
p_bins = 100
def simulation(p_bins):
    density_of_panicky_values = []
    for p in tqdm(np.arange(0, 1.01, 1/p_bins)):
        A  = Panicky_Agent(p, n_states)
        while not A.stable():
            A.update()
        density_of_panicky_values.append(A.density_of_panicky())
    return density_of_panicky_values

density_of_panicky_values = simulation(p_bins)

df = pd.DataFrame({'p': np.arange(0, 1.01, 1/p_bins), 'density_of_panicky': density_of_panicky_values})
phase_transit_point = df.where(df['density_of_panicky'] == 1).min()['p']

sns.set_context("talk")
sns.lineplot(x = 'p', y = 'density_of_panicky', data = df)
plt.axvline(x = phase_transit_point, color = 'red', linestyle = '--')
plt.title(f'p vs. density of panicky individuals\nPhase transition point at p = {phase_transit_point}')
plt.tight_layout()
savefig(f'../../simulation_figures/Q11.5/p_vs_dpi.png')
