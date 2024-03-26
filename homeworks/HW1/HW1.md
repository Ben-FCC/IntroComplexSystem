# HW1

# 4.3

1. Linear, first order, autonomous

2. Linear, third order, autonomous

3. Nonlinear, first order, autonomous

4. This equation is confused, I don’t understand the $bxt$ part. If I regard it as a constant $bx$ multiply a variable $t$, then my answer is: Nonlinear, third order, non-autonomous

5. Nonlinear, third order, non-autonomous

6. Linear, first order, autonomous

# 4.4

$$
\begin{align*}
1. \quad &x_t = x_{t-1}(1 - x_{t-1})\sin t \\
&\text{let } z_t = z_{t-1} \\
&\Rightarrow 
\begin{cases}
x_t = x_{t-1}(1 - x_{t-1})\sin(z_{t-1}) \\
z_t = z_{t-1} + 1, z_0 = 1 & \#
\end{cases} \\ 
\\ 
2. \quad &x_t = x_{t-1} + y_{t-2} - x_{t-3} \\
&\text{let }
\begin{cases}
y_{t-1} = x_{t-2} \\
z_{t-1} = x_{t-3}
\end{cases} \\
&\Rightarrow 
\begin{cases}
x_t = x_{t-1} + y_{t-1} - z_{t-1} \\
y_t = x_{t-1} \\
z_t = y_{t-1} & \#
\end{cases}
\end{align*}
$$

# 4.6

## Explain

I explored the behavior of a simple linear model across different settings for parameters **`a`** and **`b`**, specifically looking at how variations within these ranges influence the model's dynamics. I simulated the model's progression for **`a`** values in two distinct ranges: firstly, between -2.5 and -0.5 to understand the effect of negative **`a`** values, and secondly, assuming a range between 0.5 and 2.5, to observe the model's behavior with positive **`a`** values. For **`b`**, the range was kept consistent between -2 and 2 across all simulations.

## Results

### negative a

![negative_a.png](HW1%20c1785f2d4b104aa59e6fc8140520f18b/negative_a.png)

### positive a

![positive_a.png](HW1%20c1785f2d4b104aa59e6fc8140520f18b/positive_a.png)

<aside>
💡 When $a$ is negative, the results of the simulation differ significantly from the example behavior, turning into oscillation; whereas, when $a$ is positive, the nature of the simulation results is essentially the same as the behavior of the example, although the points of divergence and concentration will differ.

</aside>

## Script

```python
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
```

```python
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
```

# 4.8

## Explain

I transform this into a two-dimensional, first-order model, by introducing a system of equations that captures the essence of the Fibonacci recursion in a way that each step depends on a single previous step across two variables. This can be represented as:

$$

\begin{cases}
x_t = x_{t-1} + y_{t-1} \\
y_t = x_{t-1} \\ x_1=1 \\ y_1=1
\end{cases}

$$

Then I plotted the change of this sequence with the number of iterations, as well as its representation in Phase Space.

![Untitled](HW1%20c1785f2d4b104aa59e6fc8140520f18b/Untitled.png)

![Untitled](HW1%20c1785f2d4b104aa59e6fc8140520f18b/Untitled%201.png)

## Script

```python
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
```

```python
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
```

# 4.9

## Explain

K is set to be 3

1. Simulations with `a` > 1: In the context of population dynamics, `a` parameter *a* greater than 1 typically suggests a growth rate that leads to an increase in population over time. In these simulations, higher values of a show a marked increase in the population (represented on the y-axis) as time or iterations (UpdateTimes on the x-axis) progress. This is indicative of an exponential growth model where the rate of population increase is proportional to the current population, leading to the J-shaped curves seen in the graphs. The initial population size `x0` also plays a role; for a given `a`, a higher starting population leads to a larger population at later times. However, as a increases further, the growth becomes more rapid, and depending on the context, this could either mean reaching a higher equilibrium, continuous exponential growth, or potential overpopulation issues if carrying capacity limits are considered.
2. Simulations with `a` < 1: Here, the parameter a being less than 1 suggests a growth rate that is not sufficient to maintain the population over time. These graphs depict various scenarios where the population either declines slowly or rapidly, depending on the exact value of a and the initial population `x0`. This represents a decay model, where the population decreases over time, potentially approaching zero in some cases. This could model scenarios like population decline due to insufficient birth rates, high mortality rates, or emigration, leading to a population crash or an eventual equilibrium at a lower population size.

## Results

### small a

![Untitled](HW1%20c1785f2d4b104aa59e6fc8140520f18b/Untitled%202.png)

### big a

![Untitled](HW1%20c1785f2d4b104aa59e6fc8140520f18b/Untitled%203.png)

## Script

```python
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
```

```python
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
```