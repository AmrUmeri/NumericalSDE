# Numerical Methods for SDE

This is the repository containing the source code for the Python Package that has been developed for the bachelor thesis *Strong Schemes for the Numerical Solution of Stochastic Differential Equations* (2017). The thesis itself can be found under `.\Bachelor_Thesis.pdf`

The package in `.\Code\numerical_sde_lib` provides various (time-discrete) numerical methods for the pathwise approximation of time-independent, 1-dimensional stochastic differential equations (SDEs) with Lipschitz coefficients. The package can easily be adjusted to also allow multi-dimensional and time-dependent SDE.

Under `.\Code\examples` sample code can be found which shows how to work with the modules associated to the package.

under `.\Code\monte_carlo_estimation_convergence_order` Monte-Carlo estimates for the convergence order of the numerical methods are provided (see thesis for explanations). In particular the refinement (or "multi-resolution") algorithm for the Wiener process is iteratively applied to get approximations for the considered SDE on a finer and finer time-grid. 


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
