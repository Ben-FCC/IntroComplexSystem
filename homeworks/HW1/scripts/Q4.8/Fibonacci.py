import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class Fibonacci_model(object):
    def __init__(self):
        
        # initialize 
        self.x = 1
        self.y = 1
        self.UpdateTimes = [0]
        self.x_series = [1]
        self.y_series = [1]
    
        # set ploting theme
        sns.set_theme(context='talk', style='whitegrid')
    
    def update(self):
        
        # update the next x and y
        x_next = self.x + self.y
        y_next = self.x
        
        self.x = x_next
        self.y = y_next
        
        # save the x, y info
        self.x_series.append(self.x)
        self.y_series.append(self.y)
        
        # update info
        self.UpdateTimes = self.UpdateTimes + [self.UpdateTimes[-1]+1]
    
    def simulate(self, times: int, return_sim=False) -> pd.DataFrame:
        for _ in range(times):
            self.update()
            
        SimResults = pd.DataFrame({'UpdateTimes': self.UpdateTimes, 'x': self.x_series, 'y': self.y_series})
        if return_sim:
            return SimResults
        