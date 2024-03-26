import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class simple_linear_model(object):
    def __init__(self, x0=1, a=1, b=1):
        
        # initialize 
        self.x = x0
        self.c1 = a
        self.c0 = b
        self.UpdateTimes = [0]
        self.x_series = [x0]
    
        # set ploting theme
        sns.set_theme(context='talk', style='whitegrid')
    
    def update(self):
        
        # update the next x
        x_next = self.c1*self.x + self.c0
        self.x = x_next
        
        # save the x info
        self.x_series.append(self.x)
        
        # update info
        self.UpdateTimes = self.UpdateTimes + [self.UpdateTimes[-1]+1]
    
    def simulate(self, times: int, plot=True, return_sim=False) -> pd.DataFrame:
        for _ in range(times):
            self.update()
            
        SimResults = pd.DataFrame({'x0': self.x_series[0], 'a': self.c1, 'b': self.c0, 'UpdateTimes': self.UpdateTimes, 'x': self.x_series})
        if plot:
            sns.lineplot(x='UpdateTimes', y='x', data=SimResults)
            plt.title(f'x0: {self.x_series[0]}, a: {self.c1}, b: {self.c0}, UpdateTimes: {len(self.UpdateTimes)-1}')
            
        if return_sim:
            return SimResults