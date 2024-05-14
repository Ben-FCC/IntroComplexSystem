import matplotlib.pyplot as plt
import numpy as np
from pylab import zeros, imshow, cla, pause, argmax, random, savefig

n = 100 # size of space: n x n
#p = 0.1 # probability of initially panicky individuals

def initialize():
    global config, nextconfig
    config = zeros([n, n])
    for x in range(n):
        for y in range(n):
            for s in range(n_states-1):
                if config[x, y] == 0:
                    config[x, y] = s+1 if random() < p else 0
                else:
                    continue
                nextconfig = zeros([n, n])
    
def observe():
    global config, nextconfig
    # getting the original colormap using cm.get_cmap() function 
    orig_map=plt.cm.get_cmap('Pastel1') 
    
    # reversing the original colormap using reversed() function 
    reversed_map = orig_map.reversed() 
    cla()
    imshow(config, vmin = 0, vmax = n_states-1, cmap = reversed_map)
    
def stable():
    if np.sum(nextconfig != config):
        return False
    else:
        return True
    
def update():
    global config, nextconfig
    for x in range(n):
        for y in range(n):
            count = np.zeros(n_states)
            for s in range(n_states):
                for dx in [-r, 0, r]:
                    for dy in [-r, 0, r]:
                        if config[(x + dx) % n, (y + dy) % n] == s:
                            count[s] += 1
            nextconfig[x, y] = argmax(count)
    config, nextconfig = nextconfig, config


r = 1
for n_states in [2, 3, 4]:
    for r in [1, 2, 3]:
        for p in [0.1, 0.3, 0.5]:
            initialize()
            while not stable():
                update()
                observe()
                #pause(1)
            plt.title(f'states = {n_states}, r = {r}, p = {p}')
            savefig(f'../../simulation_figures/Q11.4/states = {n_states}, r = {r}, p = {p}.png')