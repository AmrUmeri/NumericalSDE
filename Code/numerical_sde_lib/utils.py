import numpy as np


def create_timegrid(T, n):
        """
        Returns a time-grid of [0, T] with n+1 (including 0) equi-distant discretization points.

        Parameters:
        - T: Time constant T. The total time.
        - n: Number of discretization intervals.
        """
        
        return np.linspace(0.0, T, n+1)