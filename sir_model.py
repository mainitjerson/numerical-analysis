"""
sir_model.py
============
Core SIR ODE system and numerical integrators.

Public API
----------
    sir_derivatives(S, I, R, beta, gamma, N)  → (dS, dI, dR)
    euler_sir(...)                             → (S, I, R) [, cons_err]
    rk4_sir(...)                               → (S, I, R) [, cons_err]

Notes
-----
Conservation enforcement
    After each step R is set to N - S - I (hard conservation).
    When track_conservation=True the RAW |S+I+R-N| before that
    correction is stored so genuine Euler instability remains visible.
"""

import numpy as np


# ── ODE right-hand side ───────────────────────────────────────────────────────

def sir_derivatives(S: float, I: float, R: float,
                    beta: float, gamma: float, N: float):
    """
    Right-hand side of the SIR ODE system.

        dS/dt = -β · (I/N) · S
        dI/dt =  β · (I/N) · S  −  γ · I
        dR/dt =  γ · I

    Parameters
    ----------
    S, I, R : float   current compartment values
    beta    : float   transmission rate  (day⁻¹)
    gamma   : float   recovery rate      (day⁻¹)
    N       : float   total population   (constant)

    Returns
    -------
    dS, dI, dR : float
    """
    lam = beta * I / N          # force of infection  λ = β·I/N
    dS  = -lam * S
    dI  =  lam * S - gamma * I
    dR  =  gamma * I
    return dS, dI, dR


# ── Explicit Euler ────────────────────────────────────────────────────────────

def euler_sir(S0: float, I0: float, R0_init: float,
              beta: float, gamma: float, N: float,
              dt: float, n_steps: int,
              track_conservation: bool = False):
    """
    Explicit (Forward) Euler integration of the SIR system.

    Update rule  (n → n+1):
        S_{n+1} = S_n + dt · dS_n
        I_{n+1} = I_n + dt · dI_n
        R_{n+1} = N − S_{n+1} − I_{n+1}   (conservation enforced)

    Parameters
    ----------
    S0, I0, R0_init     : float   initial conditions
    beta, gamma         : float   model parameters
    N                   : float   total population
    dt                  : float   time step (days)
    n_steps             : int     number of integration steps
    track_conservation  : bool    if True, return raw |S+I+R-N| before correction

    Returns
    -------
    S, I, R   : np.ndarray shape (n_steps+1,)
    cons_err  : np.ndarray shape (n_steps+1,)   only if track_conservation=True
    """
    S = np.zeros(n_steps + 1)
    I = np.zeros(n_steps + 1)
    R = np.zeros(n_steps + 1)
    S[0], I[0], R[0] = S0, I0, R0_init

    if track_conservation:
        cons_err    = np.zeros(n_steps + 1)
        cons_err[0] = 0.0

    for n in range(n_steps):
        dS, dI, dR = sir_derivatives(S[n], I[n], R[n], beta, gamma, N)

        S_new = S[n] + dt * dS
        I_new = I[n] + dt * dI
        R_raw = R[n] + dt * dR          # raw R before conservation fix

        if track_conservation:
            cons_err[n + 1] = abs(S_new + I_new + R_raw - N)

        S[n + 1] = S_new
        I[n + 1] = I_new
        R[n + 1] = N - S_new - I_new   # hard conservation

    if track_conservation:
        return S, I, R, cons_err
    return S, I, R


# ── Classical RK4 ────────────────────────────────────────────────────────────

def rk4_sir(S0: float, I0: float, R0_init: float,
            beta: float, gamma: float, N: float,
            dt: float, n_steps: int,
            track_conservation: bool = False):
    """
    Classical 4th-order Runge-Kutta integration of the SIR system.

    Weighted combination:
        S_{n+1} = S_n + (dt/6)(k1_S + 2k2_S + 2k3_S + k4_S)
        I_{n+1} = I_n + (dt/6)(k1_I + 2k2_I + 2k3_I + k4_I)
        R_{n+1} = N − S_{n+1} − I_{n+1}   (conservation enforced)

    Parameters / Returns : same signature as euler_sir.
    """
    S = np.zeros(n_steps + 1)
    I = np.zeros(n_steps + 1)
    R = np.zeros(n_steps + 1)
    S[0], I[0], R[0] = S0, I0, R0_init

    if track_conservation:
        cons_err    = np.zeros(n_steps + 1)
        cons_err[0] = 0.0

    def _d(s, i, r):
        return sir_derivatives(s, i, r, beta, gamma, N)

    for n in range(n_steps):
        k1S, k1I, k1R = _d(S[n],                I[n],                R[n])
        k2S, k2I, k2R = _d(S[n]+.5*dt*k1S,     I[n]+.5*dt*k1I,     R[n]+.5*dt*k1R)
        k3S, k3I, k3R = _d(S[n]+.5*dt*k2S,     I[n]+.5*dt*k2I,     R[n]+.5*dt*k2R)
        k4S, k4I, k4R = _d(S[n]+   dt*k3S,     I[n]+   dt*k3I,     R[n]+   dt*k3R)

        S_new = S[n] + (dt/6)*(k1S + 2*k2S + 2*k3S + k4S)
        I_new = I[n] + (dt/6)*(k1I + 2*k2I + 2*k3I + k4I)
        R_raw = R[n] + (dt/6)*(k1R + 2*k2R + 2*k3R + k4R)

        if track_conservation:
            cons_err[n + 1] = abs(S_new + I_new + R_raw - N)

        S[n + 1] = S_new
        I[n + 1] = I_new
        R[n + 1] = N - S_new - I_new   # hard conservation

    if track_conservation:
        return S, I, R, cons_err
    return S, I, R
