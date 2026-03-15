"""
sir_model package
=================
Numerical Analysis of the SIR Epidemic Model Using COVID-19 Data
Bulacan State University — BS Mathematics, Computer Science

Quick import
------------
    from sir_model import euler_sir, rk4_sir
    from sir_model import load_covid_data, estimate_parameters
    from sir_model import compare_methods, plot_trajectories
"""

from config               import N_PHILIPPINES, START_DATE, END_DATE, DT, BETA_LIT, GAMMA_LIT
from data_loader          import load_covid_data, prepare_initial_conditions
from sir_model            import sir_derivatives, euler_sir, rk4_sir
from parameter_estimation import create_objective, estimate_parameters
from error_metrics        import compute_error_metrics, compare_methods
from numerical_analysis   import (step_size_sensitivity, convergence_analysis,
                                   stability_analysis, print_stability_report)
from plotting             import (plot_trajectories, plot_error_analysis,
                                   plot_step_size_sensitivity, plot_convergence,
                                   plot_stability)

__all__ = [
    # config
    'N_PHILIPPINES', 'START_DATE', 'END_DATE', 'DT', 'BETA_LIT', 'GAMMA_LIT',
    # data
    'load_covid_data', 'prepare_initial_conditions',
    # model
    'sir_derivatives', 'euler_sir', 'rk4_sir',
    # estimation
    'create_objective', 'estimate_parameters',
    # metrics
    'compute_error_metrics', 'compare_methods',
    # analysis
    'step_size_sensitivity', 'convergence_analysis',
    'stability_analysis', 'print_stability_report',
    # plotting
    'plot_trajectories', 'plot_error_analysis',
    'plot_step_size_sensitivity', 'plot_convergence', 'plot_stability',
]
