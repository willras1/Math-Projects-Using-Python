import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt 
'''use scipy to solve a system of ordinary differential equations to attempt a simple model of the spread of COVID-19 using the SIR model'''

def model(p, t):
    a = 0.1 # infection rate # original .2 
    b = 0.05 # recovery rate 
    n = 330222422 
    x = p[0]
    y = p[1]
    z = p[2]
    dxdt = -a * ((x * y)/ n)
    dydt = (a * ((x * y) / n)) - (b * y)
    dzdt = b * y
    dpdt = [dxdt, dydt, dzdt]
    return dpdt

n = 330222422 
y0 = 9800 # initial infected population  
x0 = n - y0 # initial susceptible population
z0 = 0 # initial recovered population 
p0 = [x0, y0, z0] 

t = np.linspace(0, 450)

p = odeint(model, p0, t)

plt.plot(t, p[:,0], 'r', label = r'susceptible')
plt.plot(t, p[:,1], 'b', label = r'infected')
plt.plot(t, p[:,2], 'g', label = r'recorvered')
plt.legend(loc='best')
plt.xlabel('time')
plt.ylabel('population')
plt.show()