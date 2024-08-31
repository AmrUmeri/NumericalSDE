#############################################################################
# Plots the true solution and the approximation on n (and 2*n) time-points for the
# Ornstein-Uhlenbeck process
#############################################################################

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

import numpy as np
import matplotlib.pylab as plt
from numerical_sde_lib.numerical_sde_solver import NumericalSDE


# Parameter
sigma = 1.5
beta = 1.0

def a(x):
    return -beta*x
def b(x):
    return sigma

x0 = 1

#################################################
# Number of steps.
n = 200

# Initialize NumericalSDE class
numerical_sde = NumericalSDE(n)

# Wiener process, discretization of [0,T]
w = np.copy(numerical_sde.wiener)
t = np.copy(numerical_sde.timegrid)

#################################################
###############Numerical solutions###############
#################################################

# Numerical solution on n grid-points
numerical_sde.solve_sde(a,b,x0,'euler-maruyama')
Yt_euler = numerical_sde.solution
numerical_sde.solve_sde(a,b,x0,'milstein')
Yt_milstein = numerical_sde.solution
numerical_sde.solve_sde(a,b,x0,'wagner-platen')
Yt_wagnerplaten = numerical_sde.solution

# Finer versions (using refinmenent algorithm)
numerical_sde.refine_wiener()
w2 = np.copy(numerical_sde.wiener)
t2 = np.copy(numerical_sde.timegrid)

# Numerical solution on 2*n grid-points
numerical_sde.solve_sde(a,b,x0,'euler-maruyama')
Yt2_euler = numerical_sde.solution
numerical_sde.solve_sde(a,b,x0,'milstein')
Yt2_milstein = numerical_sde.solution
numerical_sde.solve_sde(a,b,x0,'wagner-platen')
Yt2_wagnerplaten = numerical_sde.solution

# Analytical solution (Ornstein-Uhlenbeck process)
temp = np.zeros(n+1)
Xt=np.zeros(n+1)
stochInt = np.zeros(n+1)

temp2 = np.zeros(2*n+1)
Xt2=np.zeros(2*n+1)
stochInt2 = np.zeros(2*n+1)

Xt[0] = x0
Xt2[0] = x0
stochInt[0] = 0
stochInt2[0] = 0
temp[0] = 0
temp2[0] = 0
for k in range(0,n):
    for j in range(0,k):
        temp[j+1] =(w[j+1]-w[j])*np.exp((-beta)*((t[k+1])-t[j]))
    stochInt[k+1] = np.sum(temp)

    Xt[k+1] = Xt[0]*np.exp((-beta)*(t[k+1])) + sigma*stochInt[k+1]

for k in range(0,2*n):
    #Xt2[k+1] = Xt[0]*np.exp((mu-sigma**2/2)*(t2[k+1]) + sigma*w2[k+1])
    for j in range(0,k):
        temp2[j+1] =(w2[j+1]-w2[j])*np.exp((-beta)*((t2[k+1])-t2[j]))
    stochInt2[k+1] = np.sum(temp2)

    Xt2[k+1] = Xt2[0]*np.exp((-beta)*(t2[k+1])) + sigma*stochInt2[k+1]

# Plot
plt.figure(1)
plt.scatter(t, Yt_euler, 1, c='b', label="Euler-method")
plt.scatter(t, Xt, 1, c='k', label="Analytical solution")
plt.fill_between(t, Xt, Yt_euler, color='b',alpha=.1, interpolate=False)
plt.legend(loc='upper left')
ax = plt.gca()
plt.text(0.025, 0.80,'n= ' + str(n), bbox=dict(facecolor='white'), transform = ax.transAxes)
plt.xlabel('t', fontsize=16)
plt.ylabel('x', fontsize=16)

plt.figure(2)
plt.scatter(t, Yt_milstein, 1, c='m', label="Milstein-method")
plt.scatter(t, Xt, 1, c='k', label="Analytical solution")
plt.fill_between(t, Xt, Yt_milstein, color='m',alpha=.1, interpolate=False)
plt.legend(loc='upper left')
ax = plt.gca()
plt.text(0.025, 0.80,'n= ' + str(n), bbox=dict(facecolor='white'), transform = ax.transAxes)
plt.xlabel('t', fontsize=16)
plt.ylabel('x', fontsize=16)

##plt.figure(3)
##plt.scatter(t, Yt_wagnerplaten, 1, c='forestgreen', label="Wagner-Platen-method")
##plt.scatter(t, Xt, 1, c='k', label="Analytical solution")
##plt.fill_between(t, Xt, Yt_wagnerplaten, color='forestgreen',alpha=.1, interpolate=False)
##plt.legend(loc='upper left')
##ax = plt.gca()
##plt.text(0.025, 0.80,'n= ' + str(n), bbox=dict(facecolor='white'), transform = ax.transAxes)
##plt.xlabel('t', fontsize=16)
##plt.ylabel('x', fontsize=16)

plt.figure(4)
plt.scatter(t2, Yt2_euler, 1, c='b', label="Euler-method")
plt.scatter(t2, Xt2, 1, c='k', label="Analytical solution")
plt.fill_between(t2, Xt2, Yt2_euler, color='b',alpha=.1, interpolate=False)
plt.legend(loc='upper left')
ax = plt.gca()
plt.text(0.025, 0.80,'n= ' + str(2*n), bbox=dict(facecolor='white'), transform = ax.transAxes)
plt.xlabel('t', fontsize=16)
plt.ylabel('x', fontsize=16)

plt.figure(5)
plt.scatter(t2, Yt2_milstein, 1, c='m', label="Milstein-method")
plt.scatter(t2, Xt2, 1, c='k', label="Analytical solution")
plt.fill_between(t2, Xt2, Yt2_milstein, color='m',alpha=.1, interpolate=False)
plt.legend(loc='upper left')
ax = plt.gca()
plt.text(0.025, 0.80,'n= ' + str(2*n), bbox=dict(facecolor='white'), transform = ax.transAxes)
plt.xlabel('t', fontsize=16)
plt.ylabel('x', fontsize=16)

##plt.figure(6)
##plt.scatter(t2, Yt2_wagnerplaten, 1, c='forestgreen', label="Wagner-Platen-method")
##plt.scatter(t2, Xt2, 1, c='k', label="Analytical solution")
##plt.fill_between(t2, Xt2, Yt2_wagnerplaten, color='forestgreen',alpha=.1, interpolate=False)
##plt.legend(loc='upper left')
#ax = plt.gca()
#plt.text(0.025, 0.80,'n= ' + str(2*n), bbox=dict(facecolor='white'), transform = ax.transAxes)
##plt.xlabel('t', fontsize=16)
##plt.ylabel('x', fontsize=16)

plt.show()
