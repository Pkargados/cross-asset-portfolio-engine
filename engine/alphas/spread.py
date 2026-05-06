#!/usr/bin/env python
# coding: utf-8
"""
alphas/spread.py — Spread (cointegration) alpha construction.

Verbatim extraction of run_baseline.py / run_engine.py step 5.
Zero logic changes.

Public API
----------
    build_spread(residual_ret, pairs, window, adf_pval, halflife_max)
        -> (spread_expanded, spread_raw)

    spread_expanded : pd.DataFrame (T, N) — asset-level normalized signal
                      (both ETF and commodity legs populated)
    spread_raw      : pd.DataFrame (T, P) — pair-level raw signals
                      (one column per pair, e.g. "XLE_CLc1")
"""

import numpy as np
import pandas as pd
from statsmodels.regression.rolling import RollingOLS
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

from engine.alphas import cs_normalize, robust_winsorize, normalize_alpha


def build_spread(
    residual_ret: pd.DataFrame,
    pairs: list,
    window: int = 120,
    adf_pval: float = 0.10,
    halflife_max: float = 30.0,
    verbose: bool = True,
) -> tuple:
    """
    Spread (cointegration) alpha with rolling OLS + rolling ADF + ECM filter.

    For each (ETF, commodity) pair:
      1. Rolling OLS hedge ratio (window days)
      2. Rolling ADF test on the residual spread (same window) — no look-ahead
      3. ECM filter: Δspread = α + ρ·spread_{t-1} — skip if ρ≥0 or hl≥halflife_max
      4. Rolling z-score: signal = -(spread_t - mu_window) / std_window

    Both legs are assigned:
      spread_expanded[etf] += signal
      spread_expanded[com] -= signal

    Parameters
    ----------
    residual_ret  : pd.DataFrame (T, N) — beta-neutralized daily returns
    pairs         : list of (etf, com) tuples
    window        : int   — rolling window for OLS, ADF, ECM (days)
    adf_pval      : float — ADF p-value threshold for cointegration
    halflife_max  : float — ECM half-life filter (days); skip if ≥ this
    verbose       : bool  — print per-pair diagnostics

    Returns
    -------
    (spread_expanded, spread_raw)
        spread_expanded : pd.DataFrame (T, N) — asset-level signal
        spread_raw      : pd.DataFrame (T, P) — pair-level raw signal
    """
    assets      = residual_ret.columns.tolist()
    pair_labels = [f"{e}_{c}" for e, c in pairs]
    spread_raw  = pd.DataFrame(index=residual_ret.index,
                                columns=pair_labels, dtype=float)

    for idx, (etf, com) in enumerate(pairs):
        if verbose:
            print(f"    Pair {idx+1}/{len(pairs)}: {etf}-{com} ...", end=" ", flush=True)

        y    = residual_ret[etf]
        X    = residual_ret[com]
        rols = RollingOLS(y, sm.add_constant(X), window=window).fit()
        spread = y - (rols.params["const"] + rols.params[com] * X)

        spread_arr = spread.values
        signal     = np.zeros(len(spread_arr))

        n_active   = 0
        half_lives = []

        for i in range(window - 1, len(spread_arr)):
            w_data  = spread_arr[i - window + 1 : i + 1]
            w_clean = w_data[~np.isnan(w_data)]
            if len(w_clean) < window // 2:
                continue
            try:
                pval = adfuller(w_clean)[1]
            except Exception:
                pval = 1.0
            if pval >= adf_pval:
                continue
            mu  = w_clean.mean()
            std = w_clean.std()
            if std < 1e-8:
                continue
            # ECM: Δspread_t = α + ρ·spread_{t-1} + ε  (within rolling window)
            # ρ < 0 → mean-reverting; half-life = -ln(2)/ln(1+ρ) days
            d_spr   = np.diff(w_clean)
            lag_spr = w_clean[:-1]
            if len(lag_spr) < 10 or lag_spr.std() < 1e-8:
                continue
            X_ecm = np.column_stack([np.ones(len(lag_spr)), lag_spr])
            try:
                coefs, _, _, _ = np.linalg.lstsq(X_ecm, d_spr, rcond=None)
            except Exception:
                continue
            rho = coefs[1]
            if rho >= 0.0 or (1.0 + rho) <= 0.0:
                continue
            half_life = -np.log(2.0) / np.log(1.0 + rho)
            if half_life >= halflife_max:
                continue
            half_lives.append(half_life)
            signal[i] = -(spread_arr[i] - mu) / std
            n_active += 1

        avg_hl = float(np.mean(half_lives)) if half_lives else float("nan")
        spread_raw[f"{etf}_{com}"] = pd.Series(signal, index=spread.index)
        if verbose:
            print(f"active={n_active} windows  avg_hl={avg_hl:.1f}d")

    # Normalize pair-level signals cross-sectionally
    spread_norm  = cs_normalize(spread_raw.astype(float))
    spread_wins  = robust_winsorize(spread_norm)
    spread_final = spread_wins

    # Expand to asset columns: ETF leg gets +signal, commodity leg gets -signal
    spread_expanded = pd.DataFrame(0.0,
                                   index=residual_ret.index,
                                   columns=residual_ret.columns)
    pair_count = {a: 0 for a in assets}
    for etf, com in pairs:
        sig = spread_final[f"{etf}_{com}"].fillna(0.0)
        spread_expanded[etf] += sig
        spread_expanded[com] -= sig
        pair_count[etf] += 1
        pair_count[com] += 1
    for asset in assets:
        if pair_count[asset] > 1:
            spread_expanded[asset] /= pair_count[asset]

    return spread_expanded, spread_raw
