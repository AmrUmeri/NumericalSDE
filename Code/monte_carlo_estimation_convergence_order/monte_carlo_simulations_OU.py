#############################################################################
# Code for figure 5.3, 5.4 in the thesis.
# Monte-Carlo estimation of the convergence order of the numerical methods 
# applied to the Ornstein-Uhlenbeck process
#############################################################################

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

import numpy as np
import matplotlib.pylab as plt
from scipy.stats import linregress
from numerical_sde_lib.numerical_sde_solver import *

# Parameter
sigma = 1.5
beta = 1.0
x0 = 1
def a(x):
    return -beta*x
def b(x):
    return sigma

############################################################################
# Number of steps. We will do the Monte-Carlo-analysis for the
# step amounts n in nval starting with n we will then always double the amount of steps
# and use the refinement algorithm
n = 8
nsize = 10
nval = np.array([n * 2**i for i in range(nsize)])
# Number of MC simulations for each step amount n
nsim = 1000

# Initialize NumericalSDE class
numerical_sde = NumericalSDE(n)

#############################################################################
#Arrays for the errors for each simulation and corresponding step n
#we are collecting the errors of the Euler, Milstein and Wagner-Platen-scheme.
error_euler = np.zeros((nsim,nsize))
error_milstein = np.zeros((nsim,nsize))
#error_wagnerplaten = np.zeros((nsim,nsize))

Yt_euler = np.zeros(nsize)
Yt_milstein = np.zeros(nsize)
#Yt_wagnerplaten = np.zeros(nsize)
Xt_T = np.zeros(nsize)
stochIntApprox = 0

##############################################################################
# Monte-Carlo-algorithm
for m in range(0,nsim):    
    T_one = numerical_sde.T_one
    for k in range(0,nsize):
        w = numerical_sde.wiener
        t = numerical_sde.timegrid

        # Approximate solution: Euler, Milstein, Wagner-Platen
        numerical_sde.solve_sde(a,b,x0,'euler-maruyama')
        Yt_euler = numerical_sde.solution
        numerical_sde.solve_sde(a,b,x0,'milstein')
        Yt_milstein = numerical_sde.solution
        numerical_sde.solve_sde(a,b,x0,'wagner-platen')
        Yt_wagnerplaten = numerical_sde.solution

        # Analytic solution
        for j in range(0,n*(2**k)):
            stochIntApprox = stochIntApprox + (w[j+1]-w[j])*np.exp((-beta)*((T_one)-t[j]))
        Xt_T = x0*np.exp((-beta)*(T_one)) + sigma*stochIntApprox

        numerical_sde.refine_wiener()
        stochIntApprox = 0
        
        # Squared-mean criterion for the error
        error_euler[m,k] = (Yt_euler[n*(2**k)] - Xt_T)**2
        error_milstein[m,k] = (Yt_milstein[n*(2**k)] - Xt_T)**2
        #error_wagnerplaten[m,k] = (Yt_wagnerplaten[n*(2**k)] - Xt_T)**2

    numerical_sde.resample_wiener(n)

    # Print progress
    print(str(float(m)/nsim*100) + '%')
##############################################################################

# Monte-Carlo estimates of the errors
mc_error_euler=np.zeros(nsize)
mc_error_milstein=np.zeros(nsize)
#mc_error_wagnerplaten=np.zeros(nsize)
for k in range(0,nsize):
    mc_error_euler[k] = np.sqrt(np.mean(error_euler[:,k]))
    mc_error_milstein[k] = np.sqrt(np.mean(error_milstein[:,k]))
    #mc_error_wagnerplaten[k] = np.sqrt(np.mean(error_wagnerplaten[:,k]))
                       
# Regression (Example: a[0] returns the slope and a[1] returns the intersect)
a = linregress(np.log2(nval), np.log2(mc_error_euler))
b = linregress(np.log2(nval), np.log2(mc_error_milstein))    
#c = linregress(np.log2(nval), np.log2(mc_error_wagnerplaten))

# log-log-plot: error estimates and step amount n
plt.figure(1)
plt.scatter(np.log2(nval), np.log2(mc_error_euler), 3, c='b', label="Euler-scheme. Slope: " + str("{0:.3f}".format(a[0])))
plt.scatter(np.log2(nval), np.log2(mc_error_milstein), 3, c='m', label="Milstein-scheme. Slope: " + str("{0:.3f}".format(b[0])))
#plt.scatter(np.log2(nval), np.log2(mc_error_wagnerplaten), 2, c='forestgreen', label="Wagner-Platen-scheme. Slope: " + str("{0:.3f}".format(c[0])))
plt.xlabel('$log_2$ n', fontsize=12)
plt.ylabel('$log_2$ error', fontsize=12)
plt.legend(loc='lower left')
# Plot regression line
plt.plot(np.log2(nval), np.log2(nval)*a[0]+a[1], c='b',alpha=0.5)
plt.plot(np.log2(nval), np.log2(nval)*b[0]+b[1], c='m',alpha=0.5)
#plt.plot(np.log2(nval), np.log2(nval)*c[0]+c[1], c='forestgreen')

# Second plot: Error analysis
plt.figure(2)
plt.plot(np.log2(nval), mc_error_euler, c='b',alpha=0.5, label="Euler-scheme")
plt.plot(np.log2(nval), mc_error_milstein, c='m',alpha=0.5, label="Milstein-scheme")
#plt.plot(np.log2(nval), mc_error_wagnerplaten, c='forestgreen', label="Wagner-Platen-scheme")
plt.xlabel('$log_2$ n', fontsize=12)
plt.ylabel('error', fontsize=12)
plt.legend(loc='upper right')

plt.show()
