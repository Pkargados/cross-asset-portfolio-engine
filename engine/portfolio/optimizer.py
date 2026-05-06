#!/usr/bin/env python
# coding: utf-8
"""
portfolio/optimizer.py — Quadratic mean-variance optimizer with turnover regularization.

Extracted verbatim from run_baseline.py::optimize_weights (Phase 3, Step 2).
No logic changes. Pure function — no global state, no I/O, no mutation of inputs.

Public API:
    chernov_weights(alpha_t, Sigma_t, x_prev, n, gamma, kappa, lambd, max_weight)
        -> np.ndarray  shape (n,)
"""

import numpy as np


def chernov_weights(alpha_t, Sigma_t, x_prev, n, gamma, kappa, lambd, max_weight):
    """
    Quadratic mean-variance optimizer with position-inertia regularization.

    Objective: maximize α'x − (γ/2)·x'Σx − (κ/2)·‖x − x_prev‖²  (with L1 cost adjustment).
    Inputs: alpha vector (E[r] units), covariance matrix Σ, previous weights x_prev.
    Closed-form solution: (γΣ + κI)·x = α + κ·x_prev − λ·sign(x_prev).
    Constraints applied post-solve: dollar neutrality, per-position limits ±max_weight.

    Parameters
    ----------
    alpha_t    : array-like (n,)   — alpha vector in E[r] units at time t
    Sigma_t    : array-like (n, n) — covariance matrix at time t
    x_prev     : array-like (n,)   — positions held at t-1
    n          : int               — number of assets
    gamma      : float             — risk-aversion coefficient
    kappa      : float             — position-inertia coefficient (penalises turnover)
    lambd      : float             — L1 penalty coefficient (transaction cost proxy)
    max_weight : float             — per-asset position limit (absolute value)

    Returns
    -------
    np.ndarray shape (n,) — new target weights, dollar-neutral, within ±max_weight.
    """
    alpha_t = np.asarray(alpha_t).reshape(-1, 1)
    x_old   = np.asarray(x_prev).reshape(-1, 1)
    Sigma   = np.asarray(Sigma_t)

    A = gamma * Sigma + kappa * np.eye(n)
    b = alpha_t + kappa * x_old - lambd * np.sign(x_old)
    x = np.linalg.solve(A, b).flatten()
    x = x - x.mean()                           # dollar neutrality
    x = np.clip(x, -max_weight, max_weight)    # position limits
    return x
