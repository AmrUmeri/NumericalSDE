#############################################################################
# Code for figures 4.1, 4.2 in the thesis.
# Figure 4.1: Generating m paths of the geometric brownian motion (GBM) on the time intervall [0,T]
# Figure 4.2: Generating m paths of the Ornstein-Uhlenbeck process (OU) on the time intervall [0,T]
#############################################################################

import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from numerical_sde_solver import NumericalSDE
from utils import create_timegrid

# Initialize parameters for the discretization and the NumericalSDE class
n = 2**8
numerical_sde = NumericalSDE(n)
t = create_timegrid(numerical_sde.T_one, n)
# Initialize number of sample paths
m = 7

#############################################################################
# Figure 4.1: Generating m paths of the geometric brownian motion (GBM) on the time intervall [0,T]
# Geometric brownian motion (SDE)
# dXt = a(Xt)dt + b(Xt)dWt
# X0 = x0
# a(x) = mu*x, b(x) = sigma*x
# mu, sigma constants
# True solution:
# Xt = X0 * exp((mu - 0.5 * sigma^2) * t + sigma * Wt)
#############################################################################

# Parameter of the SDE
sigma = 1.5
mu = 1.0
x0 = 1

# Create m discretized Wiener processes
w = np.zeros((n+1,m))
for k in range(0,m):
    numerical_sde.resample_wiener()
    w[:,k] = numerical_sde.wiener

# Generate m sample paths
Xt = np.zeros((n+1,m))
for k in range(0,m):
    Xt[0,k] = x0
    for j in range(0,n):
        Xt[j+1,k] = Xt[0,k]*np.exp((mu-sigma**2/2)*(t[k+1]) + sigma*w[j+1,k])


# Plot
for sample_path in Xt.T:
    plt.plot(t, sample_path,'r',linewidth=0.5)
plt.xlabel('t', fontsize=16)
plt.ylabel('x', fontsize=16)
plt.title(f'{m} sample paths of the Geometric Brownian Motion', fontsize=12)
plt.show()


#############################################################################
# Figure 4.2: Generating m paths of the Ornstein-Uhlenbeck process (OU) on the time intervall [0,T]
# Ornstein-Uhlenbeck process (OU)
# dXt = a(Xt)dt + b(Xt)dWt
# X0 = x0
# a(x) = -beta*x, b(x) = sigma
# beta, sigma positive constants
# True solution:
# Xt = X0 * exp(-beta * t) + (sigma / sqrt(2 * beta)) * Integral_0^t exp(-beta * (t - s)) dWs
# where Integral_0^t exp(-beta * (t - s)) dWs is a stochastic integral.
#############################################################################

# Parameter of the SDE
sigma = 1.5
beta = 1.0
x0 = 1

# Create m discretized Wiener processes
w = np.zeros((n+1,m))
for k in range(0,m):
    numerical_sde.resample_wiener()
    w[:,k] = numerical_sde.wiener

# Generate m sample paths
Xt = np.zeros((n+1,m))
stochIntApprox = np.zeros((n+1,m))
temp = np.zeros(n+1)
for s in range(0,m):
    Xt[0,s] = x0
    for k in range(0,n):
        for j in range(0,k):
            temp[j+1] =(w[j+1,s]-w[j,s])*np.exp((-beta)*((t[k+1])-t[j]))
        stochIntApprox[k+1,s] = np.sum(temp)
        temp = np.zeros(n)
        Xt[k+1,s] = Xt[0,s]*np.exp((-beta)*(t[k+1])) + sigma*stochIntApprox[k+1,s]


# Plot
for sample_path in Xt.T:
    plt.plot(t, sample_path,'r',linewidth=0.5)
plt.xlabel('t', fontsize=16)
plt.ylabel('x', fontsize=16)
plt.title(f'{m} sample paths of the Ornstein-Uhlenbeck Process', fontsize=12)
plt.show()





        


