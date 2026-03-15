"""
numerical_analysis.py
=====================
Step-size sensitivity, convergence, and stability analyses.

All three analyses use FIXED literature parameters (BETA_LIT, GAMMA_LIT)
rather than fitted values, so results reflect pure numerical behaviour
independent of dt-dependent parameter estimation noise.

Public API
----------
    step_size_sensitivity(...)   → dict  {method: {dt: cumulative_array}}
    convergence_analysis(...)    → dict  {method: {dt: MAE_vs_reference}}
    stability_analysis(...)      → dict  {method: {dt: report_dict}}
    print_stability_report(...)  → None  (console table)
"""

import numpy as np

from sir_model import euler_sir, rk4_sir


# ── Step-size sensitivity ────────────────────────────────────────────────────

def step_size_sensitivity(S0: float, I0: float, R0_init: float,
                          beta: float, gamma: float, N: float,
                          n_days_total: int, step_sizes: list) -> dict:
    """
    Run both integrators over a range of step sizes with FIXED parameters.

    Compares the final cumulative case count N−S(t) and the full trajectory
    across step sizes to show how sensitive each method is to dt choice.

    Parameters
    ----------
    S0, I0, R0_init : float   initial conditions
    beta, gamma     : float   FIXED parameters (use BETA_LIT, GAMMA_LIT)
    N               : float   total population
    n_days_total    : int     simulation duration in calendar days
    step_sizes      : list    dt values to test  e.g. [0.125, 0.25, ..., 8.0]

    Returns
    -------
    results : dict
        {
          'euler': {dt: np.ndarray  (cumulative N-S array)},
          'rk4':   {dt: np.ndarray  (cumulative N-S array)},
        }
    """
    results = {'euler': {}, 'rk4': {}}

    for dt in step_sizes:
        n_steps = int(round(n_days_total / dt))

        S_e, _, _ = euler_sir(S0, I0, R0_init, beta, gamma, N, dt, n_steps)
        S_r, _, _ = rk4_sir(  S0, I0, R0_init, beta, gamma, N, dt, n_steps)

        results['euler'][dt] = N - S_e
        results['rk4'][dt]   = N - S_r

    return results


# ── Convergence analysis ─────────────────────────────────────────────────────

def convergence_analysis(S0: float, I0: float, R0_init: float,
                         beta: float, gamma: float, N: float,
                         n_days_total: int, step_sizes: list) -> dict:
    """
    Measure how fast each method converges to the finest-dt reference.

    Strategy
    --------
    Use the smallest dt in step_sizes as the reference "true" solution.
    Compute MAE of coarser solutions against that reference.

    Expected slopes on a log-log plot:
        Euler → ~1   (1st-order convergence,  error ∝ dt¹)
        RK4   → ~4   (4th-order convergence,  error ∝ dt⁴)

    Parameters
    ----------
    step_sizes : list   MUST include the finest dt as first element
                        e.g. [0.125, 0.25, 0.5, 1.0, 2.0, 4.0]

    Returns
    -------
    convergence : dict
        {
          'euler': {dt: float  MAE vs reference},
          'rk4':   {dt: float  MAE vs reference},
        }
        Keys are all dt values except the reference (finest) dt.
    """
    step_sizes_sorted = sorted(step_sizes)
    dt_ref = step_sizes_sorted[0]
    n_ref  = int(round(n_days_total / dt_ref))

    # Reference solutions at finest dt
    S_e_ref, _, _ = euler_sir(S0, I0, R0_init, beta, gamma, N, dt_ref, n_ref)
    S_r_ref, _, _ = rk4_sir(  S0, I0, R0_init, beta, gamma, N, dt_ref, n_ref)
    cum_e_ref = N - S_e_ref
    cum_r_ref = N - S_r_ref

    convergence = {'euler': {}, 'rk4': {}}

    for dt in step_sizes_sorted[1:]:
        n_steps = int(round(n_days_total / dt))
        factor  = int(round(dt / dt_ref))

        S_e_c, _, _ = euler_sir(S0, I0, R0_init, beta, gamma, N, dt, n_steps)
        S_r_c, _, _ = rk4_sir(  S0, I0, R0_init, beta, gamma, N, dt, n_steps)
        cum_e_c = N - S_e_c
        cum_r_c = N - S_r_c

        # Subsample reference to coarse grid; clip to same length (off-by-one fix)
        ref_e_sub = cum_e_ref[::factor]
        ref_r_sub = cum_r_ref[::factor]
        ml_e = min(len(cum_e_c), len(ref_e_sub))
        ml_r = min(len(cum_r_c), len(ref_r_sub))

        convergence['euler'][dt] = float(
            np.mean(np.abs(cum_e_c[:ml_e] - ref_e_sub[:ml_e])))
        convergence['rk4'][dt]   = float(
            np.mean(np.abs(cum_r_c[:ml_r] - ref_r_sub[:ml_r])))

    return convergence


# ── Stability analysis ───────────────────────────────────────────────────────

def stability_analysis(S0: float, I0: float, R0_init: float,
                       beta: float, gamma: float, N: float,
                       n_days_total: int, step_sizes: list) -> dict:
    """
    Test whether each method remains numerically stable across step sizes.

    Instability indicators checked per run
    ---------------------------------------
    has_negative  : any S or I < 0
    has_nan       : any NaN in I
    has_explosion : any |I| > 10 × N  (10× population)
    max_cons_error: max raw |S+I+R-N| BEFORE conservation correction
                    (reveals genuine Euler blow-up — Fix #2)

    Parameters
    ----------
    step_sizes : list   include large values (e.g. 16, 32) to trigger failure

    Returns
    -------
    report : dict
        {
          'euler': {
              dt: {
                  'stable':           bool,
                  'has_negative':     bool,
                  'has_nan':          bool,
                  'has_explosion':    bool,
                  'final_cumulative': float,
                  'min_S':            float,
                  'max_cons_error':   float,
              }
          },
          'rk4': { ... same structure ... }
        }
    """
    report = {'euler': {}, 'rk4': {}}

    for dt in step_sizes:
        n_steps = int(round(n_days_total / dt))

        for method_name, solver in [('euler', euler_sir), ('rk4', rk4_sir)]:
            S, I, R, cons_err = solver(
                S0, I0, R0_init, beta, gamma, N, dt, n_steps,
                track_conservation=True)

            cum           = N - S
            has_negative  = bool(np.any(I < 0) or np.any(S < 0))
            has_nan       = bool(np.any(np.isnan(I)))
            has_explosion = bool(np.any(np.abs(I) > N * 10))
            stable        = not (has_negative or has_nan or has_explosion)

            report[method_name][dt] = {
                'stable':           stable,
                'has_negative':     has_negative,
                'has_nan':          has_nan,
                'has_explosion':    has_explosion,
                'final_cumulative': float(cum[-1]),
                'min_S':            float(np.min(S)),
                'max_cons_error':   float(np.max(cons_err)),
            }

    return report


def print_stability_report(stability_report: dict, step_sizes: list) -> None:
    """Print a formatted stability summary to the console."""
    print(f"\n  {'Method':<8} {'dt':<8} {'Stable':<10} "
          f"{'Max ConsErr':<20} {'Final Cumulative'}")
    print("  " + "-" * 68)
    for method in ['euler', 'rk4']:
        for dt in step_sizes:
            r      = stability_report[method][dt]
            status = "✓ OK"   if r['stable'] else "✗ FAIL"
            cons   = f"{r['max_cons_error']:.2e}"
            final  = f"{r['final_cumulative']:,.0f}" \
                     if not np.isnan(r['final_cumulative']) else "NaN"
            print(f"  {method.upper():<8} {dt:<8} {status:<10} "
                  f"{cons:<20} {final}")
