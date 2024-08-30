#############################################################################
# Code for figure 2.1, 2.2, 3.1 in the thesis.
# Figure 2.1: Generating m paths (samples) of a standard Wiener process on the time intervall [0,T]
# Figure 2.2: Plots a discretized Wiener process and finer versions of the same process 
#             using the refinement algorithm (see class NumericalSDE)
# Figure 3.1: Plots the Wiener process and its corresponding step process
#############################################################################

import numpy as np
import matplotlib.pyplot as plt
from numerical_sde_solver import NumericalSDE
from utils import create_timegrid

#############################################################################
# Figure 2.1: Generating m paths (samples) of a standard Wiener process on the time intervall [0,T]
#############################################################################

# Parameters for the discretization
n = 2**8

# Initialize NumericalSDE class
numerical_sde = NumericalSDE(n)

# Time discretization of [0,T] with in total n+1 grid points (including starting value 0)
t = create_timegrid(numerical_sde.T_one, n)

# Generate m discretized Wiener processes
m = 10
w = np.zeros((n+1,m))
for k in range(0,m):
    numerical_sde.resample_wiener()
    w[:,k] = numerical_sde.wiener

# Plot the Wiener processes
for sample_path in w.T:
    plt.plot(t, sample_path,'b',linewidth=0.5)
plt.xlabel('t', fontsize=16)
plt.ylabel('x', fontsize=16)
plt.show()

#############################################################################
# Figure 2.2: Plots a discretized Wiener process and finer versions of the same process 
#             using the refinement algorithm (see class NumericalSDE)
#############################################################################

# Parameters for the discretization
n = 16

# Initialize NumericalSDE class
numerical_sde = NumericalSDE(n)

wiener_0 = numerical_sde.wiener
t_0 = numerical_sde.timegrid

# Apply refinement procedure
numerical_sde.refine_wiener()
wiener_refined = numerical_sde.wiener
t_1 = numerical_sde.timegrid

# Apply refinement on the refinement
numerical_sde.refine_wiener()
wiener_refined_refined = numerical_sde.wiener
t_2 = numerical_sde.timegrid


# Plot the discretized Wiener process and refinements
plt.plot(t_0, wiener_0, 'k', linewidth=0.8)
plt.plot(t_1, wiener_refined, linewidth=0.5, c='b')
plt.plot(t_2, wiener_refined_refined, linewidth=0.5, c='b')

plt.xlabel('t', fontsize=16)
plt.ylabel('x', fontsize=16)
plt.show()



#############################################################################
# Figure 3.1: Plots the Wiener process and its corresponding step process
#############################################################################

# Parameters for the discretization
n = 16**2
n_step = 8

# Initialize NumericalSDE class
numerical_sde = NumericalSDE(n)

w = numerical_sde.wiener
t = numerical_sde.timegrid

# Create step function
w_step = np.zeros(n_step+1)
t_step = np.linspace(0.0, numerical_sde.T_one, n_step+1)


for k in range(0,n_step):
    w_step[k] = w[int(k*(n/n_step))]
w_step[n_step] = w[n-1]

# Plot a sample of the Wiener process and a corresponding step function
plt.plot(t, w,'k', linewidth=0.6)
plt.step(t_step, w_step, linewidth=1, color='b', where='post')

plt.xlabel('t', fontsize=16)
plt.ylabel('x', fontsize=16)
plt.show()


