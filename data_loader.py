"""
data_loader.py
==============
Load and validate Philippine COVID-19 cumulative confirmed cases
from the JHU CSSE repository.

Public API
----------
    load_covid_data()             → (dates, cases)
    prepare_initial_conditions()  → (S0, I0, R0_init)

Dataset notes
-------------
  Source : time_series_covid19_confirmed_global.csv
           https://github.com/CSSEGISandData/COVID-19
  Format : One row per country. Philippines has no sub-regions,
           so iloc[0] with no aggregation is correct.
  Frozen : Repository frozen 2023-03-10; study window
           (2020-03-01 – 2021-12-31) is complete and unaffected.
"""

import numpy as np
import pandas as pd

from config import N_PHILIPPINES, START_DATE, END_DATE

# ── JHU CSSE URL ──────────────────────────────────────────────────────────────
_JHU_URL = (
    "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/"
    "master/csse_covid_19_data/csse_covid_19_time_series/"
    "time_series_covid19_confirmed_global.csv"
)


def load_covid_data(url: str = _JHU_URL,
                    start: str = START_DATE,
                    end:   str = END_DATE):
    """
    Load Philippine cumulative confirmed COVID-19 cases from JHU CSSE.

    Parameters
    ----------
    url   : str   CSV source URL (default: JHU CSSE GitHub raw)
    start : str   study start date, 'YYYY-MM-DD'
    end   : str   study end   date, 'YYYY-MM-DD'

    Returns
    -------
    dates : np.ndarray[datetime64]  one entry per day in the study window
    cases : np.ndarray[float]       cumulative confirmed cases (monotone ↑)
    """
    df          = pd.read_csv(url)
    philippines = df[df['Country/Region'] == 'Philippines'].iloc[0]

    date_cols = [c for c in df.columns
                 if c not in ('Province/State', 'Country/Region', 'Lat', 'Long')]
    dates = pd.to_datetime(date_cols)
    cases = philippines[date_cols].values.astype(float)

    # ── filter to study window ────────────────────────────────────────────
    mask  = (dates >= pd.to_datetime(start)) & (dates <= pd.to_datetime(end))
    dates = dates[mask]
    cases = cases[mask]

    # ── sanity checks ─────────────────────────────────────────────────────
    diffs  = np.diff(cases)
    n_neg  = int(np.sum(diffs < 0))
    n_zero = int(np.sum(diffs == 0))

    print(f"[DATA] Loaded {len(cases)} observations  "
          f"({dates[0].date()} → {dates[-1].date()})")
    print(f"[DATA] Cumulative range : {cases[0]:.0f}  →  {cases[-1]:,.0f}")
    print(f"[DATA] Negative-increment days (data corrections) : {n_neg}")
    print(f"[DATA] Zero-increment days     (no new reporting)  : {n_zero}")

    if not np.all(diffs >= -500):
        print("[DATA] WARNING: large negative jumps detected — check data.")

    return dates.values, cases


def prepare_initial_conditions(cases_0: float, N: int = N_PHILIPPINES):
    """
    Derive SIR initial conditions from the first observed cumulative count.

    Assumptions
    -----------
    - At t = 0 active infected ≈ first reported cumulative count  (I0 = cases_0)
    - No recoveries yet at outbreak start                          (R0_init = 0)
    - All others are susceptible                                   (S0 = N - I0)

    Parameters
    ----------
    cases_0 : float   cumulative confirmed cases on day 0
    N       : int     total population

    Returns
    -------
    S0, I0, R0_init : float
    """
    I0      = float(cases_0)
    R0_init = 0.0
    S0      = N - I0 - R0_init
    return S0, I0, R0_init
