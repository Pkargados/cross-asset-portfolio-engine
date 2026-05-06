#!/usr/bin/env python
# coding: utf-8
"""
portfolio/optimizer.py — Chernov closed-form L2 optimizer.

Extracted verbatim from run_baseline.py::optimize_weights (Phase 3, Step 2).
No logic changes. Pure function — no global state, no I/O, no mutation of inputs.

Public API:
    chernov_weights(alpha_t, Sigma_t, x_prev, n, gamma, kappa, lambd, max_weight)
        -> np.ndarray  shape (n,)
"""

import numpy as np


def chernov_weights(alpha_t, Sigma_t, x_prev, n, gamma, kappa, lambd, max_weight):
    """
    Chernov closed-form L2 optimizer.

    Solves:
        min_x  (1/2) x' (γΣ) x  -  α' x  +  κ/2 ||x - x_prev||²  +  λ ||x||₁

    Closed-form solution (ignoring the L1 term in the linear system, applied
    as a soft adjustment on the RHS):
        A x = b
        A = γ Σ + κ I
        b = α + κ x_prev - λ sign(x_prev)

    Followed by:
        - dollar neutrality:  x ← x - mean(x)
        - position limits:    x ← clip(x, -max_weight, +max_weight)

    Parameters
    ----------
    alpha_t    : array-like (n,)   — alpha vector in E[r] units at time t
    Sigma_t    : array-like (n, n) — covariance matrix at time t
    x_prev     : array-like (n,)   — positions held at t-1
    n          : int               — number of assets
    gamma      : float             — risk-aversion coefficient
    kappa      : float             — position-inertia / transaction-cost coefficient
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
