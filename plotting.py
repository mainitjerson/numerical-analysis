"""
plotting.py
===========
All matplotlib visualisation functions for the SIR analysis.

Public API
----------
    plot_trajectories(...)           → Figure  (4-panel trajectory + error)
    plot_error_analysis(...)         → Figure  (3-panel error breakdown)
    plot_step_size_sensitivity(...)  → Figure  (2-panel sensitivity)
    plot_convergence(...)            → Figure  (log-log convergence)
    plot_stability(...)              → Figure  (3-panel stability)
"""

import numpy as np
import matplotlib.pyplot as plt

from config import N_PHILIPPINES


# ── Trajectories ─────────────────────────────────────────────────────────────

def plot_trajectories(dates,
                      S_euler, I_euler, R_euler,
                      S_rk4,   I_rk4,   R_rk4,
                      cases_observed,
                      N: float = N_PHILIPPINES,
                      save_path: str = None):
    """
    Four-panel comparison figure.

    Panels
    ------
    [0,0] Cumulative N−S(t) vs observed  (log scale)
    [0,1] All three SIR compartments     (RK4)
    [1,0] Cumulative prediction error
    [1,1] Euler − RK4 method divergence
    """
    cum_euler = N - S_euler
    cum_rk4   = N - S_rk4

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ── panel 1 ──────────────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(dates, cases_observed, 'k-',  lw=2,   label='Observed (JHU)', alpha=0.9)
    ax.plot(dates, cum_euler,      'b--', lw=1.5, label='Euler  (N−S)',   alpha=0.7)
    ax.plot(dates, cum_rk4,        'r-.', lw=1.5, label='RK4    (N−S)',   alpha=0.7)
    ax.set_ylabel('Cumulative Confirmed Cases', fontsize=11)
    ax.set_title('Cumulative Cases: Model vs Observed\n(N−S used as model proxy)',
                 fontsize=11, fontweight='bold')
    ax.legend(loc='upper left')
    ax.set_yscale('log')

    # ── panel 2 ──────────────────────────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(dates, S_rk4 / 1e6, 'g-', label='Susceptible', alpha=0.7)
    ax.plot(dates, I_rk4 / 1e6, 'r-', label='Infected',    alpha=0.7)
    ax.plot(dates, R_rk4 / 1e6, 'b-', label='Recovered',   alpha=0.7)
    ax.set_ylabel('Population (millions)', fontsize=11)
    ax.set_title('SIR Compartments (RK4)', fontsize=11, fontweight='bold')
    ax.legend(loc='center right')

    # ── panel 3 ──────────────────────────────────────────────────────────
    ax = axes[1, 0]
    ax.plot(dates, (cum_euler - cases_observed) / 1e3, 'b-', alpha=0.6, label='Euler error')
    ax.plot(dates, (cum_rk4   - cases_observed) / 1e3, 'r-', alpha=0.6, label='RK4   error')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_ylabel('Error (thousands)', fontsize=11)
    ax.set_xlabel('Date', fontsize=11)
    ax.set_title('Cumulative Prediction Error Over Time', fontsize=11, fontweight='bold')
    ax.legend(loc='upper left')

    # ── panel 4 ──────────────────────────────────────────────────────────
    ax = axes[1, 1]
    ax.plot(dates, (cum_euler - cum_rk4) / 1e3, color='purple', alpha=0.7)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_ylabel('Euler − RK4  (thousands)', fontsize=11)
    ax.set_xlabel('Date', fontsize=11)
    ax.set_title('Numerical Method Divergence\n(N−S Euler  minus  N−S RK4)',
                 fontsize=11, fontweight='bold')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    return fig


# ── Error analysis ────────────────────────────────────────────────────────────

def plot_error_analysis(comparison: dict, save_path: str = None):
    """
    Three-panel error breakdown figure.

    Panels
    ------
    [0] Error distribution histogram (Euler vs RK4)
    [1] Absolute error trajectory    (log scale)
    [2] MAE / RMSE / Max bar chart
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    euler_err = comparison['Euler_vs_Data']['Error_Array']
    rk4_err   = comparison['RK4_vs_Data']['Error_Array']

    # ── histogram ────────────────────────────────────────────────────────
    ax = axes[0]
    ax.hist(euler_err / 1e3, bins=50, alpha=0.5, color='blue',
            label='Euler', density=True)
    ax.hist(rk4_err   / 1e3, bins=50, alpha=0.5, color='red',
            label='RK4',   density=True)
    ax.set_xlabel('Error (thousands)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Error Distribution', fontsize=12, fontweight='bold')
    ax.legend()

    # ── absolute error trajectory ─────────────────────────────────────────
    ax = axes[1]
    ax.plot(np.abs(euler_err) / 1e3, alpha=0.6, color='blue', label='Euler')
    ax.plot(np.abs(rk4_err)   / 1e3, alpha=0.6, color='red',  label='RK4')
    ax.set_xlabel('Time (days)', fontsize=11)
    ax.set_ylabel('|Error| (thousands)', fontsize=11)
    ax.set_title('Absolute Error Trajectory', fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    ax.legend()

    # ── bar chart ─────────────────────────────────────────────────────────
    ax      = axes[2]
    metrics = ['MAE', 'RMSE', 'Max_Error']
    e_vals  = [comparison['Euler_vs_Data'][m] / 1e3 for m in metrics]
    r_vals  = [comparison['RK4_vs_Data'][m]   / 1e3 for m in metrics]
    x, w    = np.arange(len(metrics)), 0.35

    ax.bar(x - w/2, e_vals, w, color='blue', alpha=0.7, label='Euler')
    ax.bar(x + w/2, r_vals, w, color='red',  alpha=0.7, label='RK4')
    ax.set_ylabel('Error (thousands)', fontsize=11)
    ax.set_title('Error Metrics Comparison\n(vs JHU Cumulative Cases)',
                 fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    for i, (e, r) in enumerate(zip(e_vals, r_vals)):
        ax.text(i - w/2, e + 2, f'{e:.0f}', ha='center', fontsize=9)
        ax.text(i + w/2, r + 2, f'{r:.0f}', ha='center', fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    return fig


# ── Step-size sensitivity ─────────────────────────────────────────────────────

def plot_step_size_sensitivity(sensitivity_results: dict,
                                step_sizes: list,
                                n_days_total: int,
                                save_path: str = None):
    """
    Two-panel step-size sensitivity figure.

    Panels
    ------
    Left  – final cumulative value vs dt  (log x-axis)
    Right – full trajectories with colorbar  (Blue=Euler, Red=RK4)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    euler_finals = [sensitivity_results['euler'][dt][-1] / 1e6 for dt in step_sizes]
    rk4_finals   = [sensitivity_results['rk4'][dt][-1]   / 1e6 for dt in step_sizes]

    # ── final value panel ─────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(step_sizes, euler_finals, 'bo-', label='Euler', lw=2, ms=7)
    ax.plot(step_sizes, rk4_finals,   'rs-', label='RK4',   lw=2, ms=7)
    ax.set_xlabel('Step Size dt (days)', fontsize=11)
    ax.set_ylabel('Final Cumulative Cases (millions)', fontsize=11)
    ax.set_title('Step-Size Sensitivity: Final Value', fontsize=12, fontweight='bold')
    ax.legend()
    ax.set_xscale('log')

    # ── trajectory panel with colorbar ────────────────────────────────────
    ax     = axes[1]
    cmap_b = plt.cm.Blues
    cmap_r = plt.cm.Reds
    norm   = plt.Normalize(vmin=0, vmax=len(step_sizes) - 1)

    for k, dt in enumerate(step_sizes):
        t_e = np.linspace(0, n_days_total, len(sensitivity_results['euler'][dt]))
        t_r = np.linspace(0, n_days_total, len(sensitivity_results['rk4'][dt]))
        ax.plot(t_e, sensitivity_results['euler'][dt] / 1e6,
                '--', color=cmap_b(norm(k)), alpha=0.8)
        ax.plot(t_r, sensitivity_results['rk4'][dt]   / 1e6,
                '-',  color=cmap_r(norm(k)), alpha=0.8)

    sm = plt.cm.ScalarMappable(cmap=cmap_b, norm=norm)
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, ticks=range(len(step_sizes)))
    cb.set_ticklabels([f'dt={dt}' for dt in step_sizes])
    cb.set_label('Step sizes  (Blue=Euler, Red=RK4)', fontsize=9)

    ax.set_xlabel('Time (days)', fontsize=11)
    ax.set_ylabel('Cumulative Cases (millions)', fontsize=11)
    ax.set_title('Full Trajectories Across Step Sizes', fontsize=12, fontweight='bold')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    return fig


# ── Convergence ───────────────────────────────────────────────────────────────

def plot_convergence(convergence_results: dict, save_path: str = None):
    """
    Log-log convergence plot with empirical slope annotations.

    Expected slopes:
        Euler ≈ 1   (O dt¹)
        RK4   ≈ 4   (O dt⁴)
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    dts_e  = sorted(convergence_results['euler'].keys())
    dts_r  = sorted(convergence_results['rk4'].keys())
    errs_e = [convergence_results['euler'][dt] for dt in dts_e]
    errs_r = [convergence_results['rk4'][dt]   for dt in dts_r]

    ax.loglog(dts_e, errs_e, 'bo-', label='Euler', lw=2, ms=8)
    ax.loglog(dts_r, errs_r, 'rs-', label='RK4',   lw=2, ms=8)

    dt_arr = np.array(dts_e, dtype=float)
    ax.loglog(dt_arr, errs_e[0] * (dt_arr / dts_e[0]) ** 1,
              'b--', alpha=0.4, label='O(dt¹) reference')
    ax.loglog(dt_arr, errs_r[0] * (dt_arr / dts_r[0]) ** 4,
              'r--', alpha=0.4, label='O(dt⁴) reference')

    if len(dts_e) >= 2:
        slope_e = np.polyfit(np.log(dts_e), np.log(errs_e), 1)[0]
        slope_r = np.polyfit(np.log(dts_r), np.log(errs_r), 1)[0]
        ax.text(0.05, 0.85, f'Euler empirical slope ≈ {slope_e:.2f}',
                transform=ax.transAxes, color='blue', fontsize=11)
        ax.text(0.05, 0.78, f'RK4   empirical slope ≈ {slope_r:.2f}',
                transform=ax.transAxes, color='red',  fontsize=11)

    ax.set_xlabel('Step Size dt (days)', fontsize=11)
    ax.set_ylabel('MAE vs. Reference Solution (N−S)', fontsize=11)
    ax.set_title('Convergence Analysis: Euler vs. RK4', fontsize=12, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    return fig


# ── Stability ─────────────────────────────────────────────────────────────────

def plot_stability(stability_report: dict,
                   step_sizes: list,
                   N: float = N_PHILIPPINES,
                   save_path: str = None):
    """
    Three-panel stability figure.

    Panels
    ------
    Left   – colour-coded OK / FAIL stability map
    Centre – final cumulative value vs dt
    Right  – max raw conservation error |S+I+R−N| vs dt
    """
    fig, axes   = plt.subplots(1, 3, figsize=(18, 5))
    methods     = ['euler', 'rk4']
    method_lbls = ['Euler', 'RK4']

    # ── stability map ─────────────────────────────────────────────────────
    ax = axes[0]
    ax.set_xlim(-0.5, len(step_sizes) - 0.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xticks(range(len(step_sizes)))
    ax.set_xticklabels([f'dt={dt}' for dt in step_sizes], rotation=45, ha='right')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(method_lbls)
    ax.set_title('Stability Map', fontsize=12, fontweight='bold')

    for i, method in enumerate(methods):
        for j, dt in enumerate(step_sizes):
            stable = stability_report[method][dt]['stable']
            rect   = plt.Rectangle((j-.4, i-.4), .8, .8,
                                    color='green' if stable else 'red', alpha=0.7)
            ax.add_patch(rect)
            ax.text(j, i, 'OK' if stable else 'FAIL',
                    ha='center', va='center',
                    fontsize=10, fontweight='bold', color='white')

    # ── final cumulative value ────────────────────────────────────────────
    ax = axes[1]
    for method, lbl, col, mk in [('euler','Euler','blue','o'),
                                   ('rk4',  'RK4',  'red', 's')]:
        finals = [stability_report[method][dt]['final_cumulative'] / 1e6
                  for dt in step_sizes]
        ax.plot(step_sizes, finals, f'{col[0]}{mk}-', label=lbl, lw=2, ms=7)
    ax.set_xlabel('Step Size dt (days)', fontsize=11)
    ax.set_ylabel('Final Cumulative Cases (millions)', fontsize=11)
    ax.set_title('Final Cumulative Value vs dt', fontsize=12, fontweight='bold')
    ax.legend()
    ax.set_xscale('log')

    # ── conservation error ────────────────────────────────────────────────
    ax = axes[2]
    for method, lbl, col, mk in [('euler','Euler','blue','o'),
                                   ('rk4',  'RK4',  'red', 's')]:
        cons = [stability_report[method][dt]['max_cons_error']
                for dt in step_sizes]
        ax.plot(step_sizes, cons, f'{col[0]}{mk}-', label=lbl, lw=2, ms=7)
    ax.set_xlabel('Step Size dt (days)', fontsize=11)
    ax.set_ylabel('Max |S+I+R−N|  (raw, before correction)', fontsize=11)
    ax.set_title('Conservation Error vs dt', fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    return fig
