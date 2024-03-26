import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class population_growth_model(object):
    def __init__(self, x0=1, a=1, K=3):
        
        # initialize 
        self.x = x0
        self.a = a
        self.K = K
        self.UpdateTimes = [0]
        self.x_series = [x0]
    
        # set ploting theme
        sns.set_theme(context='talk', style='whitegrid')
    
    def update(self):
        
        # update the next x
        x_next = ((self.a - 1)/(self.K))*self.x + self.a
        self.x = x_next
        
        # save the x info
        self.x_series.append(self.x)
        
        # update info
        self.UpdateTimes = self.UpdateTimes + [self.UpdateTimes[-1]+1]
    
    def simulate(self, times: int, plot=True, return_sim=False) -> pd.DataFrame:
        for _ in range(times):
            self.update()
            
        SimResults = pd.DataFrame({'x0': self.x_series[0], 'a': self.a, 'K': self.K, 'UpdateTimes': self.UpdateTimes, 'x': self.x_series})
        if plot:
            sns.lineplot(x='UpdateTimes', y='x', data=SimResults)
            plt.title(f'x0: {self.x_series[0]}, a: {self.a}, K: {self.K}, UpdateTimes: {len(self.UpdateTimes)-1}')
            
        if return_sim:
            return SimResults