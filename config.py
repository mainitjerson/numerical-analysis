"""
config.py
=========
Global constants for the SIR model numerical analysis.

Usage
-----
    from config import N_PHILIPPINES, START_DATE, END_DATE, DT, BETA_LIT, GAMMA_LIT
"""

# ── Population ────────────────────────────────────────────────────────────────
N_PHILIPPINES = 108_000_000          # 2020 Philippine census

# ── Study window ─────────────────────────────────────────────────────────────
START_DATE = '2020-03-01'
END_DATE   = '2021-12-31'

# ── Baseline integration step ────────────────────────────────────────────────
DT = 1.0                             # day

# ── Fixed literature parameters (used in sensitivity & convergence) ──────────
# beta=0.30, gamma=0.10 → R0=3.0 (plausible early-pandemic estimate)
# These are intentionally NOT the fitted values so that sensitivity and
# convergence analyses reflect pure numerical behaviour, not fitting noise.
BETA_LIT  = 0.30
GAMMA_LIT = 0.10
