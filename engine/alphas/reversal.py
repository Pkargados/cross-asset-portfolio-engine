#!/usr/bin/env python
# coding: utf-8
"""
alphas/reversal.py — Short-term reversal alpha.

Logic is verbatim from run_engine.py / run_baseline.py. Zero changes.
"""

import pandas as pd
from engine.alphas import normalize_alpha


def build_reversal(residual_ret: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Short-term reversal: negative rolling sum of residual returns.

    Parameters
    ----------
    residual_ret : pd.DataFrame
        Beta-neutralized daily returns.
    window : int
        Rolling window in days (default 5).

    Returns
    -------
    pd.DataFrame
        Normalized (CS z-score + MAD winsorize) reversal alpha.
    """
    rev_raw = -residual_ret.rolling(window).sum()
    return normalize_alpha(rev_raw)
