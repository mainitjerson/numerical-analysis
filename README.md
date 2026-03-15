# Numerical Analysis of the SIR Epidemic Model

**Bulacan State University — BS Mathematics, Computer Science**

A Python project that applies numerical methods (Explicit Euler and classical Runge–Kutta 4) to the **SIR epidemic model**, using **Philippine COVID-19** cumulative confirmed cases from the Johns Hopkins University (JHU) CSSE dataset. The workflow covers data loading, parameter estimation, forward integration, error analysis, step-size sensitivity, convergence, and stability.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Mathematical Background](#mathematical-background)
- [Module Reference](#module-reference)
- [Installation & Usage](#installation--usage)
- [Pipeline Summary](#pipeline-summary)
- [Outputs](#outputs)
- [Key Findings](#key-findings)

---

## Overview

The SIR model divides the population into three compartments:

- **S** — Susceptible  
- **I** — Infected (active cases)  
- **R** — Recovered  

The dynamics are described by a system of ordinary differential equations (ODEs). This project:

1. Loads JHU Philippine cumulative confirmed cases over a configurable date range.
2. Estimates transmission rate (β) and recovery rate (γ) by fitting the model’s **cumulative incidence** \(N - S(t)\) to observed cumulative cases (not \(I(t)\)).
3. Integrates the ODEs with **Explicit Euler** and **RK4**.
4. Compares both methods to data and to each other (MAE, RMSE, MAPE, etc.).
5. Runs **step-size sensitivity**, **convergence**, and **stability** analyses using fixed literature parameters.

All logic lives in dedicated modules; `main.py` only orchestrates the pipeline.

---

## Project Structure

```
Numerical-Analysis/
├── README.md              # This file
├── main.py                # Entry point — runs the full pipeline
├── config.py              # Global constants (population, dates, dt, BETA_LIT, GAMMA_LIT)
├── data_loader.py         # JHU data fetch and initial conditions
├── sir_model.py           # SIR ODE and Euler / RK4 integrators
├── parameter_estimation.py # Nelder–Mead fitting (β, γ) vs N−S(t)
├── error_metrics.py       # MAE, RMSE, MAPE, method comparison
├── numerical_analysis.py   # Step-size sensitivity, convergence, stability
├── plotting.py            # All matplotlib figures
├── __init__.py            # Package exports for convenience imports
├── .gitignore
└── results/               # Generated figures (created on first run)
    ├── sir_trajectories.png
    ├── error_analysis.png
    ├── step_size_sensitivity.png
    ├── convergence_analysis.png
    └── stability_analysis.png
```

---

## Mathematical Background

### SIR ODEs

\[
\frac{dS}{dt} = -\beta \frac{I}{N} S, \quad
\frac{dI}{dt} = \beta \frac{I}{N} S - \gamma I, \quad
\frac{dR}{dt} = \gamma I
\]

- **β** (beta): transmission rate (per day).  
- **γ** (gamma): recovery rate (per day).  
- **N**: total population (constant).  
- **λ = β I/N**: force of infection.

Conservation: \(S + I + R = N\) at all times. The code enforces this by setting \(R_{n+1} = N - S_{n+1} - I_{n+1}\) after each step; optional tracking of the raw \(S+I+R-N\) before correction is used for stability analysis.

### Cumulative incidence and data

- **Observed data**: JHU *cumulative* confirmed cases.  
- **Model proxy**: \(N - S(t)\) (everyone who has left the susceptible compartment), i.e. cumulative incidence.  
- **Not used for fitting**: \(I(t)\) (current infected only).  

Parameter estimation minimizes the sum of squared errors between \(N - S(t)\) and the JHU cumulative series.

### Numerical methods

- **Explicit Euler**: \(S_{n+1} = S_n + \Delta t \cdot dS_n\), etc., with R corrected for conservation. First-order accurate: error \(\propto \Delta t\).  
- **RK4**: Classical 4th-order Runge–Kutta; same conservation fix. Fourth-order accurate: error \(\propto \Delta t^4\).

---

## Module Reference

### `config.py`

Central place for constants:

- **N_PHILIPPINES**: Population (e.g. 108,000,000).  
- **START_DATE**, **END_DATE**: Study window (e.g. `2020-03-01` to `2021-12-31`).  
- **DT**: Baseline time step in days (e.g. 1.0).  
- **BETA_LIT**, **GAMMA_LIT**: Fixed “literature” parameters used only in step-size sensitivity, convergence, and stability (so those analyses reflect numerics, not fitting).

---

### `data_loader.py`

- **`load_covid_data(url, start, end)`**  
  - Reads JHU global time series CSV from URL (default: GitHub raw).  
  - Filters to Philippines and to `[start, end]`.  
  - Returns `(dates, cases)` — NumPy arrays of dates and cumulative cases.  
  - Prints basic validation (observation count, range, negative/zero increments).

- **`prepare_initial_conditions(cases_0, N)`**  
  - Sets \(I_0 = \texttt{cases\_0}\), \(R_0 = 0\), \(S_0 = N - I_0 - R_0\).  
  - Used to initialize the SIR model from the first observed cumulative count.

---

### `sir_model.py`

- **`sir_derivatives(S, I, R, beta, gamma, N)`**  
  - Returns \((dS/dt, dI/dt, dR/dt)\).

- **`euler_sir(S0, I0, R0_init, beta, gamma, N, dt, n_steps, track_conservation=False)`**  
  - Explicit Euler integration.  
  - Returns `(S, I, R)` arrays; if `track_conservation=True`, also returns raw conservation error before the \(R = N - S - I\) fix.

- **`rk4_sir(...)`**  
  - Same signature and return convention as `euler_sir`, using the classical RK4 scheme.

---

### `parameter_estimation.py`

- **`create_objective(cases_observed, S0, I0, R0_init, N, dt, n_steps, method)`**  
  - Builds a scalar objective: sum of squared errors between \(N - S(t)\) and `cases_observed`.  
  - Parameters are log-transformed (\(\beta = \exp(p_0)\), \(\gamma = \exp(p_1)\)) so \(\beta, \gamma > 0\).  
  - `method` is `'rk4'` or `'euler'`.

- **`estimate_parameters(..., method='rk4', n_restarts=5)`**  
  - Minimizes the objective with **Nelder–Mead** and multiple random restarts to reduce risk of local minima.  
  - Returns \(\hat\beta\), \(\hat\gamma\), \(\hat R_0 = \hat\beta/\hat\gamma\), and the best `scipy.optimize` result.

---

### `error_metrics.py`

- **`compute_error_metrics(model_cumulative, cases_observed)`**  
  - Computes MAE, RMSE, Max Error, MAPE (only for days with >1000 cases), error std, skewness, and raw/absolute error arrays.

- **`compare_methods(cum_euler, cum_rk4, cases_observed)`**  
  - Compares Euler and RK4 cumulative predictions to each other (max/mean difference, correlation) and to data (full error metrics for each).

---

### `numerical_analysis.py`

Uses **fixed** β and γ (e.g. `BETA_LIT`, `GAMMA_LIT`) so results reflect numerical behaviour, not parameter fitting.

- **`step_size_sensitivity(S0, I0, R0_init, beta, gamma, N, n_days_total, step_sizes)`**  
  - Runs Euler and RK4 for each `dt` in `step_sizes` over `n_days_total` days.  
  - Returns `{'euler': {dt: cum_array}, 'rk4': {dt: cum_array}}` where `cum_array = N - S`.

- **`convergence_analysis(...)`**  
  - Uses the finest `dt` in `step_sizes` as reference.  
  - For coarser `dt`, computes MAE of \(N-S(t)\) vs the reference (subsampled).  
  - Returns MAE per method and per `dt` (expect Euler ~O(dt), RK4 ~O(dt⁴) in log-log).

- **`stability_analysis(...)`**  
  - For each method and `dt`, runs with `track_conservation=True`.  
  - Checks for negative S/I, NaNs, “explosion” (|I| > 10N), and max raw conservation error.  
  - Returns a report dict and supports **`print_stability_report(report, step_sizes)`** for console output.

---

### `plotting.py`

All functions accept optional `save_path` and use `seaborn-v0_8-whitegrid` (set in `main.py`).

- **`plot_trajectories(...)`** — 2×2: cumulative N−S vs observed (log), SIR compartments (RK4), prediction error over time, Euler−RK4 divergence.  
- **`plot_error_analysis(comparison)`** — Error distribution, |error| trajectory (log), MAE/RMSE/Max bar chart.  
- **`plot_step_size_sensitivity(...)`** — Final cumulative vs dt (log x), trajectories by dt (Euler dashed, RK4 solid).  
- **`plot_convergence(convergence_results)`** — Log-log MAE vs dt with O(dt¹) and O(dt⁴) reference lines and empirical slopes.  
- **`plot_stability(stability_report, step_sizes)`** — Stability map (OK/FAIL), final cumulative vs dt, max conservation error vs dt.

---

## Installation & Usage

### Requirements

- Python 3.8+  
- NumPy  
- Pandas  
- Matplotlib  
- SciPy  

Install with:

```bash
pip install numpy pandas matplotlib scipy
```

Optional: use a virtual environment and/or a `requirements.txt`:

```
numpy
pandas
matplotlib
scipy
```

### Running the pipeline

From the project root:

```bash
python main.py
```

This will:

1. Load Philippine COVID-19 data from JHU (or fail with a clear error if the URL is unreachable).  
2. Prepare initial conditions and estimate β, γ (and R₀) with RK4 and Euler.  
3. Integrate forward with both methods.  
4. Compute and print error metrics and method comparison.  
5. Create all figures and save them under `results/` (directory created if missing).  
6. Run step-size sensitivity, convergence, and stability analyses and print summaries.  
7. Print a short summary and conclusions.

No command-line arguments are required; all configuration is in `config.py` and the code.

---

## Pipeline Summary

| Step | Description |
|------|-------------|
| 1 | **Data loading** — JHU Philippines cumulative cases; validation info printed. |
| 2 | **Parameter estimation** — Nelder–Mead on N−S(t) vs data (RK4 primary, Euler check). |
| 3 | **Forward integration** — Euler and RK4 with estimated parameters. |
| 4 | **Error analysis** — compare_methods; MAE, RMSE, MAPE, method divergence. |
| 5 | **Visualisation** — trajectories and error analysis figures. |
| 6 | **Step-size sensitivity** — fixed β, γ; multiple dt; final cumulative and trajectories. |
| 7 | **Convergence** — MAE vs reference solution; log-log slopes. |
| 8 | **Stability** — stability report and conservation error vs dt. |
| 9 | **Summary** — dataset note, parameter estimates, errors, and main conclusions. |

---

## Outputs

- **Console**: Progress by step, parameter estimates, error metrics, stability table, and final summary.  
- **`results/`**:  
  - `sir_trajectories.png` — Model vs data and SIR compartments.  
  - `error_analysis.png` — Error distributions and metrics.  
  - `step_size_sensitivity.png` — Effect of dt on final value and trajectories.  
  - `convergence_analysis.png` — Convergence order (Euler ≈ 1, RK4 ≈ 4).  
  - `stability_analysis.png` — Stability map and conservation error.

---

## Key Findings (from the pipeline summary)

1. **N−S(t)** is the correct model quantity to compare to JHU cumulative cases; **I(t)** is not.  
2. Euler’s conservation error grows with dt and is visible in the stability panel.  
3. Convergence analysis confirms **Euler ≈ O(dt¹)** and **RK4 ≈ O(dt⁴)**.  
4. Both methods are stable for dt ≤ 1 day; Euler can fail for large dt (e.g. 8–16 days).  
5. The main error source is **model structure** (constant β, no under-reporting/vaccination, etc.), not only numerics.

---

## Quick import (package use)

If you use the project as a package (e.g. run from a parent directory or install it), you can do:

```python
from sir_model import (
    euler_sir, rk4_sir,
    load_covid_data, prepare_initial_conditions,
    estimate_parameters, compare_methods,
    plot_trajectories, plot_error_analysis,
    step_size_sensitivity, convergence_analysis,
    stability_analysis, print_stability_report,
)
from config import N_PHILIPPINES, START_DATE, END_DATE, DT, BETA_LIT, GAMMA_LIT
```

The `__init__.py` re-exports these symbols for convenience.

---

## License and data

- **Code**: Use and modify as needed for academic or personal use; attribute Bulacan State University if required by your institution.  
- **Data**: JHU CSSE COVID-19 Data ([GitHub](https://github.com/CSSEGISandData/COVID-19)); follow their terms of use and citation.
