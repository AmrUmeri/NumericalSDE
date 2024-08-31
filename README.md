# Numerical Methods for SDE

The package in `.\Code\numerical_sde_lib` provides various (time-discrete) numerical methods for solving time-independent, 1-dimensional stochastic differential equations (SDEs). The package can easily be adjusted to also allow multi-dimensional and time-dependent SDE.

Under `.\Code\examples` sample code can be found which shows how to work with the modules associated to the package.


## Content of `numerical_sde_lib`

- **Discretization of the Wiener Process**: Simple techniques for simulating sample paths of the discretized Wiener process.
- **Refinement Algorithm for the Wiener Process**: Algorithm based on the Brownian Bridge for refining a given discretized Wiener path, providing a finer resolution. The algorithm can be reiterated. (See the thesis for detailed explanations.)
- **Numerical Methods**:
  - **Euler-Maruyama**: Numerical method of convergence order 0.5 based on the usual Euler-Method for ODE.
  - **Milstein**: Numerical method of convergence order 1.0, constructed by truncating the first stochastic Taylor expansion.
  - **Wagner-Platen**: Numerical method of convergence order 1.5, constructed by truncating the second stochastic Taylor expansion.


## Installing the Package

You can install the package locally (e.g., in a Python virtual environment) by navigating to the `.\Code\numerical_sde_lib` directory and running:

```bash
pip install .
