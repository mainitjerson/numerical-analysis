"""
parameter_estimation.py
=======================
Least-squares parameter estimation for the SIR model using
Nelder-Mead optimisation with multiple random restarts.

Public API
----------
    create_objective(...)      → objective function for scipy.minimize
    estimate_parameters(...)   → (beta_hat, gamma_hat, R0_hat, result)

Key fix
-------
    The objective compares N − S(t) (cumulative incidence) against
    JHU cumulative confirmed cases — NOT I(t) which is only currently
    active infected individuals.
"""

import numpy as np
from scipy.optimize import minimize

from sir_model import euler_sir, rk4_sir


def create_objective(cases_observed: np.ndarray,
                     S0: float, I0: float, R0_init: float,
                     N: float, dt: float, n_steps: int,
                     method: str = 'rk4'):
    """
    Build a least-squares objective function for SIR parameter estimation.

    Objective  :  Σ [ (N − S(t)) − cases_observed(t) ]²

    N − S(t) is the cumulative incidence (everyone who ever left the
    susceptible pool), which matches JHU's cumulative confirmed cases.
    Parameters are log-transformed to enforce β, γ > 0 during search.

    Parameters
    ----------
    cases_observed : np.ndarray   JHU cumulative confirmed cases
    S0, I0, R0_init: float        initial conditions
    N              : float        total population
    dt             : float        integration step size
    n_steps        : int          number of integration steps
    method         : 'rk4' | 'euler'

    Returns
    -------
    objective : callable  f([log_beta, log_gamma]) → SSE (float)
    """
    solver = rk4_sir if method == 'rk4' else euler_sir

    def objective(params):
        beta  = np.exp(params[0])
        gamma = np.exp(params[1])

        S, _, _ = solver(S0, I0, R0_init, beta, gamma, N, dt, n_steps)

        cumulative_model = N - S                   # correct proxy
        error            = cumulative_model - cases_observed
        return float(np.sum(error ** 2))

    return objective


def estimate_parameters(cases_observed: np.ndarray,
                        S0: float, I0: float, R0_init: float,
                        N: float, dt: float, n_steps: int,
                        method: str = 'rk4',
                        n_restarts: int = 5):
    """
    Estimate β and γ via Nelder-Mead with multiple random restarts.

    Multiple restarts guard against local minima in the non-convex
    log-likelihood surface of the SIR model.

    Parameters
    ----------
    cases_observed : np.ndarray
    S0, I0, R0_init: float
    N              : float
    dt             : float
    n_steps        : int
    method         : 'rk4' | 'euler'
    n_restarts     : int     number of random restarts (default 5)

    Returns
    -------
    beta_hat  : float   estimated transmission rate   (day⁻¹)
    gamma_hat : float   estimated recovery rate       (day⁻¹)
    R0_hat    : float   basic reproduction number  β / γ
    result    : scipy OptimizeResult   best optimisation run
    """
    obj = create_objective(cases_observed, S0, I0, R0_init,
                           N, dt, n_steps, method)

    best_result = None
    best_value  = np.inf

    for i in range(n_restarts):
        # restart 0: near literature values; others: random
        x0 = np.log([0.3, 0.1]) if i == 0 else \
             np.log(np.random.uniform([0.05, 0.03], [1.0, 0.3]))

        result = minimize(obj, x0, method='Nelder-Mead',
                          options={'maxiter': 5000,
                                   'xatol':   1e-8,
                                   'fatol':   1e-8})
        if result.fun < best_value:
            best_value  = result.fun
            best_result = result

    beta_hat  = float(np.exp(best_result.x[0]))
    gamma_hat = float(np.exp(best_result.x[1]))
    R0_hat    = beta_hat / gamma_hat

    print(f"[{method.upper()}] β̂={beta_hat:.4f} day⁻¹  "
          f"γ̂={gamma_hat:.4f} day⁻¹  "
          f"R₀={R0_hat:.3f}  "
          f"SSE={best_result.fun:.3e}  "
          f"iters={best_result.nit}")

    return beta_hat, gamma_hat, R0_hat, best_result
