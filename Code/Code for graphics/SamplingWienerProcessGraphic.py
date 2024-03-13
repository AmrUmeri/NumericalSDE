#############################################################################
###Code for figure 2.1
###Generating m paths of a standard Wiener process on the time intervall [0,T]
###SamplingWienerProcess.py
###Python 2.7
#############################################################################
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from NumericalSDE import *
      
#Parameters for the discretization
n =2**8
#Time discredized [0,T] in total n+1 elements (including starting value 0)
t = timegrid(n)
#m discretized Wiener processes
m = 10
w = np.zeros((n+1,m))
for k in range(0,m):
    w[:,k] = wiener(n)

#Plot the Wiener Processes
for sample_path in w.T:
    plt.plot(t, sample_path,'b',linewidth=0.5)
plt.xlabel('t', fontsize=16)
plt.ylabel('x', fontsize=16)
plt.show()
