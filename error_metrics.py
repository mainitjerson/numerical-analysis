"""
error_metrics.py
================
Error metrics and method comparison utilities.

Public API
----------
    compute_error_metrics(model_cumulative, cases_observed)  → dict
    compare_methods(cum_euler, cum_rk4, cases_observed)      → dict

MAPE note
---------
    MAPE is computed ONLY over days where cumulative cases > 1000
    to avoid near-zero denominator distortion from early outbreak days
    (Philippine data starts at 3 cases on 2020-03-01).
"""

import numpy as np


def compute_error_metrics(model_cumulative: np.ndarray,
                          cases_observed:   np.ndarray) -> dict:
    """
    Compute error metrics between model N−S(t) and JHU observed cases.

    Metrics
    -------
    MAE       : Mean Absolute Error
    RMSE      : Root Mean Square Error
    Max_Error : Maximum Absolute Error
    MAPE      : Mean Absolute Percentage Error  (days > 1000 cases only)
    Error_Std : Standard deviation of errors
    Error_Skew: Skewness of error distribution
    Error_Array     : raw signed errors  (model − observed)
    Abs_Error_Array : absolute errors

    Parameters
    ----------
    model_cumulative : np.ndarray   N − S(t) from integrator
    cases_observed   : np.ndarray   JHU cumulative confirmed cases

    Returns
    -------
    dict with all metrics listed above
    """
    error     = model_cumulative - cases_observed
    abs_error = np.abs(error)

    mae     = float(np.mean(abs_error))
    rmse    = float(np.sqrt(np.mean(error ** 2)))
    max_err = float(np.max(abs_error))

    # MAPE: meaningful only when case counts are non-trivially small
    mask_mape = cases_observed > 1000
    mape = float(np.mean(abs_error[mask_mape] /
                          cases_observed[mask_mape]) * 100) \
           if mask_mape.sum() > 0 else float('nan')

    error_std = float(np.std(error))
    error_skew = float(
        np.mean(((error - np.mean(error)) / error_std) ** 3)
    ) if error_std > 0 else 0.0

    return {
        'MAE':             mae,
        'RMSE':            rmse,
        'Max_Error':       max_err,
        'MAPE':            mape,
        'Error_Std':       error_std,
        'Error_Skew':      error_skew,
        'Error_Array':     error,
        'Abs_Error_Array': abs_error,
    }


def compare_methods(cum_euler:     np.ndarray,
                    cum_rk4:       np.ndarray,
                    cases_observed: np.ndarray) -> dict:
    """
    Side-by-side comparison of Euler and RK4 cumulative predictions.

    Parameters
    ----------
    cum_euler       : N − S_euler(t)
    cum_rk4         : N − S_rk4(t)
    cases_observed  : JHU cumulative confirmed cases

    Returns
    -------
    dict with keys:
        Max_Method_Diff    : max  |Euler − RK4|
        Mean_Method_Diff   : mean |Euler − RK4|
        Method_Correlation : Pearson r between Euler and RK4 outputs
        Euler_vs_Data      : compute_error_metrics result for Euler
        RK4_vs_Data        : compute_error_metrics result for RK4
    """
    method_diff = np.abs(cum_euler - cum_rk4)

    return {
        'Max_Method_Diff':    float(np.max(method_diff)),
        'Mean_Method_Diff':   float(np.mean(method_diff)),
        'Method_Correlation': float(np.corrcoef(cum_euler, cum_rk4)[0, 1]),
        'Euler_vs_Data':      compute_error_metrics(cum_euler, cases_observed),
        'RK4_vs_Data':        compute_error_metrics(cum_rk4,   cases_observed),
    }
