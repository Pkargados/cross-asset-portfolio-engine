#!/usr/bin/env python
# coding: utf-8
"""
alphas/ — Alpha construction modules.

Shared helpers (cs_normalize, robust_winsorize, normalize_alpha) are defined
here so momentum.py and spread.py can import from a single location without
depending on run_baseline.py or run_engine.py.

Logic is verbatim from run_baseline.py / run_engine.py. Zero changes.
"""

import numpy as np
import pandas as pd


def cs_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score per row. Ignores NaN."""
    mu  = df.mean(axis=1, skipna=True)
    std = df.std(axis=1, skipna=True).clip(lower=1e-8)
    return df.sub(mu, axis=0).div(std, axis=0)


def robust_winsorize(df: pd.DataFrame, z: float = 3.0) -> pd.DataFrame:
    """MAD-based cross-sectional winsorization."""
    med = df.median(axis=1)
    mad = df.sub(med, axis=0).abs().median(axis=1).clip(lower=1e-8)
    return df.clip(lower=med - z * mad, upper=med + z * mad, axis=0)


def normalize_alpha(raw: pd.DataFrame) -> pd.DataFrame:
    """CS z-score → MAD winsorize."""
    return robust_winsorize(cs_normalize(raw))
