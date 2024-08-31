#############################################################################
# Main Module: Numerical Methods for (time-independent, 1-dimensional) SDE
#
# This package can be:
# - Imported into other scripts.
# - Installed locally via pip.
#
# Features include:
# - Discretization of the Wiener process
# - Refinement algorithm for the Wiener process (see thesis for details)
# - Numerical methods:
#   - Euler-Maruyama
#   - Milstein
#   - Wagner-Platen
#############################################################################


from math import sqrt
import numpy as np
from scipy.stats import norm
from numerical_sde_lib.utils import create_timegrid


class NumericalSDE: 
    """
    A class that provides methods for numerically solving 1-dimensional time-independent 
    stochastic differential equations (SDEs).

    This class includes:
    - Methods for discretizing the Wiener process and time intervals.
    - Refinement algorithms for the Wiener process.
    - Numerical solvers for SDEs: Euler-Maruyama, Milstein, and Wagner-Platen schemes.
    """

    def __init__(self, n_time_grid=1, random_seed=69):
        """
        Initialize the numerical SDE solver.

        Parameters:
        - n_time_grid: Number of grids in the interval [0, T] for the approximation (Not counting 0)
        - random_seed: Random seed used for the generation of discrete Wiener processes
        """
        self.T_one = 1.0
        self.mu = None
        self.sigma = None
        self.x0 = None
        self.random_state = np.random.RandomState(random_seed)
        self.wiener = self.discretize_wiener(n_time_grid)
        self.timegrid = create_timegrid(self.T_one, n_time_grid)
        self.solution = None

        # Map approximation schemes to corresponding methods.
        self.method_map = {
            'euler-maruyama': self.approximate_sde_euler_maruyama,
            'milstein': self.approximate_sde_milstein,
            'wagner-platen': self.approximate_sde_wagner_platen
        }


    def discretize_wiener(self, n):
        """
        Generates the increments for the Wiener process starting with 0 at time t=0.

        Parameters:
        - n: The number of discretization points (excluding the initial point at t=0) of the time interval [0, T].
        """
        dt = float(self.T_one)/n

        # Generate n independent samples from a normal distribution N(0, dt)
        # Insert 0 at the beginning to represent the starting value at t=0.
        increments = np.insert(norm.rvs(size=n, scale=sqrt(dt), random_state=self.random_state),0,0)

        # This computes the Wiener process by forming the cumulative sum of
        # the random samples and returns its values
        return np.cumsum(increments)
    
    def refine_wiener(self):
        """
        Returns a refinement of the discretized Wiener process.
        The resulting discrete Wiener process will have 2n+1 (including 0) discretization points.
        Allows to have better approximations of the SDE.

        The method is inserting new points between the existing original points via a
        brownian bridge procedure. 

        In addition, the method creates a new timegrid as well with 2n+1 equi-distant discretization points.
        """
        w = self.wiener
        n = w.size-1
        dt = float(self.T_one)/n

        rval=np.empty(w.size*2-1)

        for k in range((w.size-1)):
            rval[2*k] = w[k]
            rval[2*k+1] = norm.rvs(size=1, loc=(w[k]+w[k+1])/2, scale=sqrt(dt/4), random_state=self.random_state)

        rval[w.size*2-2] = w[w.size-1]
            
        self.wiener = rval
        self.timegrid = create_timegrid(self.T_one, 2*n)


    def resample_wiener(self, n=None):
        """
        Generates a new sample for the discretized Wiener process starting with 0 at time t=0.
        """
        if n==None:
            w = self.wiener
            n = w.size-1
        self.wiener = self.discretize_wiener(n)
        self.timegrid = create_timegrid(self.T_one, n)


    def solve_sde(self, func_mu, func_sigma, x0, method='milstein'):
        """
        Solves the SDE using the specified method.
        Returns the approximated values on the time-grid.

        Parameters:
        - func_mu: The function defining the drift term of the SDE, of the form a(x).
        - func_sigma: The function defining the noise term of the SDE, of the form b(x).
        - x0: Initial condition.
        - method: The method for the approximation ('milstein', 'euler-maruyama', 'wagner-platen'). Default is 'milstein'.
        """

        self.mu = func_mu
        self.sigma = func_sigma
        self.x0 = x0

        if method in self.method_map:
            self.solution = self.method_map[method]()
        else:
            raise ValueError(f"Unknown method: {method}")
        


    def approximate_sde_euler_maruyama(self):
        """
        Implements the Euler-Maruyama method for SDEs.
        """

        w = self.wiener
        a = self.mu
        b = self.sigma

        n = w.size-1
        dt = float(self.T_one)/n

        Xval = np.zeros(n+1)
        Xval[0] = self.x0
        for k in range(0,n):
            a_val           =   a(Xval[k])
            b_val           =   b(Xval[k])

            Xval[k+1] = Xval[k] + a_val*dt + b_val*(w[k+1]-w[k])

        return Xval

    
    def approximate_sde_milstein(self):
        """
        Implements the Milstein method for SDEs.
        """

        w = self.wiener
        a = self.mu
        b = self.sigma
        
        n = w.size-1
        dt = float(self.T_one)/n

        Xval = np.zeros(n+1)
        Xval[0] = self.x0
        for k in range(0,n):
            eps = 1e-5 * max(1.0, abs(Xval[k]))
            a_val           =   a(Xval[k])
            b_val           =   b(Xval[k])
            b_dv_val        =  float((b(Xval[k] + eps) - b(Xval[k])) / eps)
            
            Xval[k+1] = (Xval[k] + a_val*dt + b_val*(w[k+1]-w[k])
                        + float(1)/2*b_val*b_dv_val*((w[k+1]-w[k])**2-dt))
        return Xval



    def approximate_sde_wagner_platen(self):
        """
        Implements the Wagner-Platen method for SDEs.
        """

        w = self.wiener
        a = self.mu
        b = self.sigma

        n = w.size-1
        dt = float(self.T_one)/n 

        Z = np.zeros(n)
        for k in range(0,n):
            Z[k] = 0.5*dt**1.5*((w[k+1]-w[k])/sqrt(dt) + norm.rvs(random_state=self.random_state)/sqrt(3))
        
        Xval = np.zeros(n+1)
        Xval[0] = self.x0
        for k in range(0,n):
            eps = 1e-5 * max(1.0, abs(Xval[k]))
            a_val           =   a(Xval[k])
            b_val           =   b(Xval[k])
            a_dv_val        =   float((a(Xval[k] + eps) - a(Xval[k])) / eps)
            b_dv_val        =   float((b(Xval[k] + eps) - b(Xval[k])) / eps)
            a_dvdv_val      =   float((a(Xval[k] + eps) - 2 * a_val + a(Xval[k] - eps)) / eps**2)
            b_dvdv_val      =   float((b(Xval[k] + eps) - 2 * b_val + b(Xval[k] - eps)) / eps**2)
                    
            Xval[k+1] = (Xval[k] + a_val*dt + b_val*(w[k+1]-w[k])
                        + float(1)/2*b_val*b_dv_val*((w[k+1]-w[k])**2-dt)
                        + a_dv_val*b_val*Z[k]
                        + float(1)/2*(a_val*a_dv_val + float(1)/2*b_val**2*a_dvdv_val)*dt**2
                        + (a_val*b_dv_val + float(1)/2*b_val**2*b_dvdv_val)*((w[k+1]-w[k])*dt-Z[k])
                        + float(1)/2*b_val*(b_val*b_dvdv_val + (b_dv_val)**2)*(float(1)/3*(w[k+1]-w[k])**2-dt)*(w[k+1]-w[k]))
        return Xval


