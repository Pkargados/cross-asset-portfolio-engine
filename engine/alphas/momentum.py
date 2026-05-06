#!/usr/bin/env python
# coding: utf-8
"""
alphas/momentum.py — Momentum alpha construction.

Verbatim extraction of run_baseline.py / run_engine.py step 4.
Zero logic changes.

Public API
----------
    build_momentum(residual_ret, skip, window) -> pd.DataFrame
        Returns CS-normalized, MAD-winsorized momentum alpha.
        Shape: same as residual_ret.
"""

import pandas as pd

from engine.alphas import normalize_alpha


def build_momentum(
    residual_ret: pd.DataFrame,
    skip: int = 5,
    window: int = 60,
) -> pd.DataFrame:
    """
    Momentum alpha: skip-adjusted rolling sum of beta-neutralized returns.

    Parameters
    ----------
    residual_ret : pd.DataFrame (T, N) — beta-neutralized daily returns
    skip         : int — lookback skip (days) to avoid short-term reversal
    window       : int — rolling sum window (days)

    Returns
    -------
    pd.DataFrame (T, N) — CS z-scored + MAD-winsorized momentum signal
    """
    mom_raw = residual_ret.shift(skip).rolling(window).sum()
    return normalize_alpha(mom_raw)
