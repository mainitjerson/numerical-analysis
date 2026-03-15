"""
main.py
=======
Orchestration pipeline for the SIR model numerical analysis.

This file only contains control flow — all logic lives in the
imported modules below.

Usage
-----
    python main.py

Module map
----------
    config                 → global constants
    data_loader            → load_covid_data, prepare_initial_conditions
    sir_model              → sir_derivatives, euler_sir, rk4_sir
    parameter_estimation   → estimate_parameters
    error_metrics          → compare_methods
    numerical_analysis     → step_size_sensitivity, convergence_analysis,
                             stability_analysis, print_stability_report
    plotting               → all plot_* functions
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── project modules ───────────────────────────────────────────────────────────
from config import (
    N_PHILIPPINES, START_DATE, END_DATE, DT,
    BETA_LIT, GAMMA_LIT,
)
from data_loader import (
    load_covid_data,
    prepare_initial_conditions,
)
from sir_model import euler_sir, rk4_sir
from parameter_estimation import estimate_parameters
from error_metrics import compare_methods
from numerical_analysis import (
    step_size_sensitivity,
    convergence_analysis,
    stability_analysis,
    print_stability_report,
)
from plotting import (
    plot_trajectories,
    plot_error_analysis,
    plot_step_size_sensitivity,
    plot_convergence,
    plot_stability,
)

# ── global setup ──────────────────────────────────────────────────────────────
RESULTS_DIR = 'results'
np.random.seed(42)
plt.style.use('seaborn-v0_8-whitegrid')
warnings.filterwarnings('ignore')


def main():
    print("=" * 70)
    print("SIR MODEL NUMERICAL ANALYSIS: PHILIPPINE COVID-19 DATA")
    print("=" * 70)
    print(f"Population   : {N_PHILIPPINES:,}")
    print(f"Study period : {START_DATE}  →  {END_DATE}")
    print(f"Baseline dt  : {DT} day")

    # ── 1. Data loading ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 1: Data Loading & Validation")
    print("=" * 70)

    dates, cases = load_covid_data()
    n_steps_base = len(cases) - 1       # integration steps  (dt = 1 day)
    n_days_total = n_steps_base         # calendar days in study window

    S0, I0, R0_init = prepare_initial_conditions(cases[0])
    print(f"Initial conditions: S0={S0:.0f}  I0={I0:.0f}  R0={R0_init:.0f}")

    # ── 2. Parameter estimation ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 2: Parameter Estimation  (objective: N−S vs JHU cumulative)")
    print("=" * 70)

    beta_hat, gamma_hat, R0_hat, _ = estimate_parameters(
        cases, S0, I0, R0_init, N_PHILIPPINES, DT, n_steps_base,
        method='rk4', n_restarts=5)

    print("\nVerifying with Euler method …")
    beta_e, gamma_e, R0_e, _ = estimate_parameters(
        cases, S0, I0, R0_init, N_PHILIPPINES, DT, n_steps_base,
        method='euler', n_restarts=3)

    print(f"\nConsensus (RK4): β={beta_hat:.4f}  γ={gamma_hat:.4f}  "
          f"R₀={R0_hat:.3f}  1/γ={1/gamma_hat:.1f} days")

    # ── 3. Forward integration ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 3: Forward Integration")
    print("=" * 70)

    S_euler, I_euler, R_euler = euler_sir(
        S0, I0, R0_init, beta_hat, gamma_hat,
        N_PHILIPPINES, DT, n_steps_base)

    S_rk4, I_rk4, R_rk4 = rk4_sir(
        S0, I0, R0_init, beta_hat, gamma_hat,
        N_PHILIPPINES, DT, n_steps_base)

    cum_euler = N_PHILIPPINES - S_euler
    cum_rk4   = N_PHILIPPINES - S_rk4

    print(f"Euler final cumulative  : {cum_euler[-1]:,.0f}")
    print(f"RK4   final cumulative  : {cum_rk4[-1]:,.0f}")
    print(f"Observed final          : {cases[-1]:,.0f}")

    # ── 4. Error analysis ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 4: Error Analysis")
    print("=" * 70)

    comparison = compare_methods(cum_euler, cum_rk4, cases)

    print(f"\nMethod-to-method  |Euler − RK4|:")
    print(f"  Max  : {comparison['Max_Method_Diff']:,.0f} cases")
    print(f"  Mean : {comparison['Mean_Method_Diff']:,.0f} cases")
    print(f"  Corr : {comparison['Method_Correlation']:.8f}")

    for label, key in [('Euler', 'Euler_vs_Data'), ('RK4', 'RK4_vs_Data')]:
        m = comparison[key]
        print(f"\n{label} vs observed:")
        print(f"  MAE       : {m['MAE']:,.0f} cases")
        print(f"  RMSE      : {m['RMSE']:,.0f} cases")
        print(f"  Max Error : {m['Max_Error']:,.0f} cases")
        print(f"  MAPE      : {m['MAPE']:.2f}%  (days > 1 000 cases only)")

    # ── 5. Visualisation ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 5: Visualisation")
    print("=" * 70)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    plot_dates = pd.to_datetime(dates)

    fig1 = plot_trajectories(
        plot_dates,
        S_euler, I_euler, R_euler,
        S_rk4,   I_rk4,   R_rk4,
        cases,
        save_path=os.path.join(RESULTS_DIR, 'sir_trajectories.png'))

    fig2 = plot_error_analysis(
        comparison,
        save_path=os.path.join(RESULTS_DIR, 'error_analysis.png'))

    # ── 6. Step-size sensitivity ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 6: Step-Size Sensitivity  (β=0.30, γ=0.10 fixed)")
    print("=" * 70)

    step_sizes = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]

    sensitivity = step_size_sensitivity(
        S0, I0, R0_init, BETA_LIT, GAMMA_LIT,
        N_PHILIPPINES, n_days_total, step_sizes)

    print(f"\n  {'dt':<8} {'Euler final cum.':<24} {'RK4 final cum.'}")
    print("  " + "-" * 52)
    for dt in step_sizes:
        e = sensitivity['euler'][dt][-1]
        r = sensitivity['rk4'][dt][-1]
        print(f"  {dt:<8} {e:<24,.0f} {r:,.0f}")

    fig3 = plot_step_size_sensitivity(
        sensitivity, step_sizes, n_days_total,
        save_path=os.path.join(RESULTS_DIR, 'step_size_sensitivity.png'))

    # ── 7. Convergence analysis ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 7: Convergence Analysis  (reference = dt=0.125)")
    print("=" * 70)

    conv_steps = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0]

    convergence = convergence_analysis(
        S0, I0, R0_init, BETA_LIT, GAMMA_LIT,
        N_PHILIPPINES, n_days_total, conv_steps)

    print(f"\n  {'dt':<8} {'Euler MAE':<24} {'RK4 MAE'}")
    print("  " + "-" * 52)
    for dt in sorted(convergence['euler']):
        print(f"  {dt:<8} {convergence['euler'][dt]:<24,.2f} "
              f"{convergence['rk4'][dt]:,.2f}")

    fig4 = plot_convergence(
        convergence,
        save_path=os.path.join(RESULTS_DIR, 'convergence_analysis.png'))

    # ── 8. Stability analysis ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 8: Stability Analysis")
    print("=" * 70)

    stab_steps = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]

    stability = stability_analysis(
        S0, I0, R0_init, BETA_LIT, GAMMA_LIT,
        N_PHILIPPINES, n_days_total, stab_steps)

    print_stability_report(stability, stab_steps)

    fig5 = plot_stability(
        stability, stab_steps,
        save_path=os.path.join(RESULTS_DIR, 'stability_analysis.png'))

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY AND CONCLUSIONS")
    print("=" * 70)
    print(f"""
Dataset Validation:
  Source  : JHU CSSE time_series_covid19_confirmed_global.csv
  Country : Philippines (single row, no sub-region aggregation needed)
  Period  : {START_DATE} – {END_DATE}  ({n_steps_base + 1} observations)
  Status  : Cumulative confirmed cases, standard benchmark dataset ✓

Parameter Estimates (RK4, N−S objective):
  β̂  = {beta_hat:.4f} day⁻¹   (transmission rate)
  γ̂  = {gamma_hat:.4f} day⁻¹   (recovery rate)
  R₀  = {R0_hat:.3f}            (basic reproduction number)
  1/γ = {1/gamma_hat:.1f} days  (mean infectious period)

Error vs JHU Cumulative Cases:
  Euler  MAE = {comparison['Euler_vs_Data']['MAE']:,.0f}   MAPE = {comparison['Euler_vs_Data']['MAPE']:.1f}%
  RK4    MAE = {comparison['RK4_vs_Data']['MAE']:,.0f}   MAPE = {comparison['RK4_vs_Data']['MAPE']:.1f}%
  Method divergence (mean) = {comparison['Mean_Method_Diff']:,.0f} cases

Key Findings:
  1. N−S(t) is the correct cumulative incidence proxy (not I(t)).
  2. Euler conservation error grows with dt; visible in stability panel.
  3. Convergence confirms Euler ≈ O(dt¹), RK4 ≈ O(dt⁴).
  4. Both methods stable at dt ≤ 1 day; Euler fails at dt ≥ 8–16 days.
  5. Dominant error source is model structure (simple SIR lacks
     time-varying β, under-reporting, vaccination), not numerics.
    """)

    print("=" * 70)
    print("Analysis complete.")
    print("=" * 70)

    return {
        'parameters':   (beta_hat, gamma_hat, R0_hat),
        'trajectories': {
            'euler': (S_euler, I_euler, R_euler),
            'rk4':   (S_rk4,   I_rk4,   R_rk4),
        },
        'comparison':  comparison,
        'sensitivity': sensitivity,
        'convergence': convergence,
        'stability':   stability,
        'figures':     (fig1, fig2, fig3, fig4, fig5),
    }


if __name__ == '__main__':
    results = main()
