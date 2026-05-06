#!/usr/bin/env python
# coding: utf-8
"""
report.py — Standardized strategy performance reporting module.

Computes metrics after each pipeline phase and saves to JSON.
Called by run_baseline.py (and future run_phaseN.py scripts).

Inputs:
    pnl          -- pd.Series: portfolio PnL (daily or weekly)
    returns_df   -- pd.DataFrame: asset return matrix (T x N)
    alpha_dict   -- {"Rev": df, "Mom": df, "Spread": df}
    standalone   -- {"Rev": {"pnl": series, "sharpe": float}, ...}  (optional)
    weights_df   -- pd.DataFrame: portfolio weights                  (optional)

Outputs:
    reports/report_<phase_name>.json
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy import stats


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def _detect_periods_per_year(pnl: pd.Series) -> int:
    """Return 52 (weekly) or 252 (daily) based on median gap between observations."""
    if len(pnl) < 3 or not hasattr(pnl.index, "to_series"):
        return 252
    median_gap = pnl.index.to_series().diff().median()
    if pd.isna(median_gap):
        return 252
    return 52 if median_gap.days >= 5 else 252


def compute_performance(pnl: pd.Series) -> dict:
    """
    Annualized performance from a PnL series.

    Auto-detects daily (252) vs weekly (52) periodicity from index gaps.
    Safe against negative cumulative return (returns NaN instead of complex).
    """
    pnl = pnl.dropna()
    n   = len(pnl)
    if n < 2:
        return {"error": "insufficient data", "n_periods": n}

    freq = _detect_periods_per_year(pnl)
    total_return = float((1 + pnl).prod() - 1)

    # Safe annualization: (1 + total_return) <= 0 means total loss or worse
    base = 1.0 + total_return
    if base <= 0:
        ann_return = float("nan")
    else:
        ann_return = float(base ** (freq / n) - 1)

    ann_vol = float(pnl.std() * np.sqrt(freq))
    sharpe  = ann_return / ann_vol if (ann_vol > 1e-12 and not np.isnan(ann_return)) else float("nan")

    cumret      = (1 + pnl).cumprod()
    running_max = cumret.cummax()
    dd          = (cumret - running_max) / running_max
    max_dd      = float(dd.min())

    calmar = (ann_return / abs(max_dd)
              if (abs(max_dd) > 1e-12 and not np.isnan(ann_return))
              else float("nan"))

    return {
        "total_return":          round(total_return, 6),
        "annualized_return":     None if np.isnan(ann_return) else round(ann_return, 6),
        "annualized_volatility": round(ann_vol, 6),
        "sharpe_ratio":          None if np.isnan(sharpe)     else round(sharpe, 4),
        "calmar_ratio":          None if np.isnan(calmar)     else round(calmar, 4),
        "max_drawdown":          round(max_dd, 4),
        "n_periods":             int(n),
        "start_date":            str(pnl.index[0].date())  if hasattr(pnl.index[0],  "date") else str(pnl.index[0]),
        "end_date":              str(pnl.index[-1].date()) if hasattr(pnl.index[-1], "date") else str(pnl.index[-1]),
    }


# ---------------------------------------------------------------------------
# Signal quality — Information Coefficient
# ---------------------------------------------------------------------------

def compute_ic(alpha_df: pd.DataFrame, returns_df: pd.DataFrame) -> dict:
    """
    Rank IC (Spearman) between alpha signal at t and 1-period forward returns.

    IC_t = spearmanr(alpha_t, return_{t+1}) across assets.

    Returns mean IC, IC std, IC t-stat, and count.
    """
    common_idx  = alpha_df.index.intersection(returns_df.index)
    fwd_returns = returns_df.shift(-1)

    ic_values = []
    for date in common_idx[:-1]:   # drop last row (no forward return)
        a    = alpha_df.loc[date].dropna()
        r    = fwd_returns.loc[date].dropna()
        cols = a.index.intersection(r.index)
        if len(cols) < 4:
            continue
        corr, _ = stats.spearmanr(a[cols], r[cols])
        if not np.isnan(corr):
            ic_values.append(float(corr))

    ic_arr = np.array(ic_values)
    if len(ic_arr) < 5:
        return {"mean_ic": None, "ic_std": None, "ic_tstat": None, "ic_count": len(ic_arr)}

    mean_ic = float(np.mean(ic_arr))
    ic_std  = float(np.std(ic_arr, ddof=1))
    se      = ic_std / np.sqrt(len(ic_arr))
    tstat   = mean_ic / se if se > 1e-12 else float("nan")

    return {
        "mean_ic":  round(mean_ic, 6),
        "ic_std":   round(ic_std, 6),
        "ic_tstat": round(tstat, 4),
        "ic_count": int(len(ic_arr)),
    }


# ---------------------------------------------------------------------------
# Stability — rolling Sharpe and turnover
# ---------------------------------------------------------------------------

def compute_stability(
    pnl:        pd.Series,
    weights_df: pd.DataFrame = None,
    window:     int = 60,
) -> dict:
    """
    Rolling Sharpe (annualized, rolling `window`-period window) and portfolio turnover.
    """
    pnl = pnl.dropna()

    freq = _detect_periods_per_year(pnl)
    roll_mean   = pnl.rolling(window).mean()
    roll_std    = pnl.rolling(window).std()
    roll_sharpe = (roll_mean / roll_std.clip(lower=1e-12)) * np.sqrt(freq)
    avg_rs      = float(roll_sharpe.dropna().mean())

    pct_positive = float((pnl > 0).mean())

    avg_turnover = None
    if weights_df is not None and len(weights_df) > 1:
        avg_turnover = round(float(weights_df.diff().abs().sum(axis=1).mean()), 6)

    return {
        "avg_rolling_sharpe":    round(avg_rs, 4),
        "pct_positive_periods":  round(pct_positive, 4),
        "avg_turnover":          avg_turnover,
        "rolling_sharpe_window": int(window),
    }


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def compute_diagnostics(
    weights_df:  pd.DataFrame,
    alpha_dict:  dict,
    returns_df:  pd.DataFrame,
) -> dict:
    """
    Additional diagnostics beyond core performance metrics.

    1. Gross exposure  — mean(sum |w_t|) over time
    2. Spread stats    — distribution of spread alpha signal (non-zero values)
    3. IC per alpha    — mean IC + std, keyed by alpha name
    """
    diag = {}

    # 1. Gross exposure
    if weights_df is not None and len(weights_df) > 0:
        gross_exp_series = weights_df.abs().sum(axis=1)
        diag["gross_exposure"] = {
            "mean":   round(float(gross_exp_series.mean()), 6),
            "median": round(float(gross_exp_series.median()), 6),
            "max":    round(float(gross_exp_series.max()), 6),
        }
    else:
        diag["gross_exposure"] = None

    # 2. Spread signal distribution (non-zero entries only)
    spread_df = alpha_dict.get("Spread", alpha_dict.get("spread", None))
    if spread_df is not None:
        vals = spread_df.values.flatten()
        vals = vals[np.isfinite(vals) & (vals != 0.0)]
        if len(vals) > 0:
            diag["spread_stats"] = {
                "mean":    round(float(np.mean(vals)),   6),
                "std":     round(float(np.std(vals)),    6),
                "min":     round(float(np.min(vals)),    6),
                "max":     round(float(np.max(vals)),    6),
                "n_nonzero": int(len(vals)),
                "pct_nonzero": round(float(len(vals) / spread_df.size), 4),
            }
        else:
            diag["spread_stats"] = {"n_nonzero": 0, "pct_nonzero": 0.0}
    else:
        diag["spread_stats"] = None

    # 3. IC per alpha (mean + std)
    ic_diag = {}
    for name, alpha_df in alpha_dict.items():
        ic_result = compute_ic(alpha_df, returns_df)
        ic_diag[name] = {
            "mean_ic": ic_result.get("mean_ic"),
            "ic_std":  ic_result.get("ic_std"),
        }
    diag["ic"] = ic_diag

    return diag


# ---------------------------------------------------------------------------
# PnL Attribution
# ---------------------------------------------------------------------------

def compute_attribution(standalone: dict) -> dict:
    """
    Summarize per-alpha standalone backtest results.

    `standalone` must be:
        {"Rev":    {"pnl": pd.Series, "sharpe": float, ...},
         "Mom":    {...},
         "Spread": {...}}
    """
    out = {}
    for name, res in standalone.items():
        entry = {}
        if "pnl" in res and res["pnl"] is not None:
            perf = compute_performance(res["pnl"])
            entry["sharpe_ratio"]      = perf.get("sharpe_ratio")
            entry["total_return"]      = perf.get("total_return")
            entry["annualized_return"] = perf.get("annualized_return")
            entry["max_drawdown"]      = perf.get("max_drawdown")
        for k, v in res.items():
            if k != "pnl":
                entry[k] = v
        out[name] = entry
    return out


# ---------------------------------------------------------------------------
# Regime statistics
# ---------------------------------------------------------------------------

def compute_regime_stats(
    regime_df:    pd.DataFrame,
    scale_series: pd.Series,
    dcc_converged: bool = None,
) -> dict:
    """
    Summarize DCC-GARCH regime detection outputs for the JSON report.

    Parameters
    ----------
    regime_df    : DataFrame with columns rho_t, disp_t, spike_t, ewma_vol_t,
                   spread_stab (produced by regime_detection.compute_regime_signals)
    scale_series : Series with values in [0.1, 1.0] (per rebalancing date)
    dcc_converged: bool or None — DCC optimization convergence flag

    Returns
    -------
    dict suitable for json.dump
    """
    out = {}

    if dcc_converged is not None:
        out["dcc_converged"] = bool(dcc_converged)

    if regime_df is not None and len(regime_df) > 0:
        rho = regime_df["rho_t"].dropna()
        out["rho_mean"]  = round(float(rho.mean()),  6) if len(rho) else None
        out["rho_std"]   = round(float(rho.std()),   6) if len(rho) else None

        disp = regime_df["disp_t"].dropna()
        out["disp_mean"] = round(float(disp.mean()), 6) if len(disp) else None

        spike = regime_df["spike_t"].dropna()
        out["n_spikes"]  = int(spike.sum())
        out["spike_pct"] = round(float(spike.mean()), 6) if len(spike) else None

        ewma = regime_df["ewma_vol_t"].dropna() if "ewma_vol_t" in regime_df.columns else pd.Series(dtype=float)
        out["ewma_vol_mean"] = round(float(ewma.mean()), 6) if len(ewma) else None

        stab = regime_df["spread_stab"].dropna() if "spread_stab" in regime_df.columns else pd.Series(dtype=float)
        out["spread_stab_mean"] = round(float(stab.mean()), 6) if len(stab) else None

    if scale_series is not None and len(scale_series) > 0:
        sc = scale_series.dropna()
        out["scale_mean"]          = round(float(sc.mean()), 6)
        out["scale_min"]           = round(float(sc.min()),  6)
        out["n_periods_reduced"]   = int((sc < 1.0).sum())
        out["pct_periods_reduced"] = round(float((sc < 1.0).mean()), 6)

    return out


# ---------------------------------------------------------------------------
# Master report generator
# ---------------------------------------------------------------------------

def generate_report(
    pnl:          pd.Series,
    returns_df:   pd.DataFrame,
    alpha_dict:   dict,
    standalone:   dict           = None,
    weights_df:   pd.DataFrame   = None,
    phase_name:   str            = "Phase1",
    description:  str            = "",
    output_dir:   str            = "reports",
    regime_df:    pd.DataFrame   = None,
    scale_series: pd.Series      = None,
    dcc_converged: bool          = None,
) -> dict:
    """
    Build a full report dict and save to reports/report_<phase_name>.json.

    Parameters
    ----------
    pnl         : pd.Series — combined portfolio PnL (net of transaction costs)
    returns_df  : pd.DataFrame — T x N asset returns (used for IC)
    alpha_dict  : {"Rev": df, "Mom": df, "Spread": df} — alpha signal frames
    standalone  : {"Rev": {"pnl": series}, ...} — optional per-alpha backtest results
    weights_df  : pd.DataFrame — T x N weight matrix (used for turnover + gross exposure)
    phase_name  : str — label for this report (e.g. "Phase1")
    description : str — human-readable description of changes
    output_dir  : str — directory to write JSON

    Returns
    -------
    dict — full report (also written to disk)
    """
    print(f"\nGenerating report: {phase_name}")

    report = {
        "metadata": {
            "phase":        phase_name,
            "description":  description,
            "generated_at": datetime.utcnow().isoformat() + "Z",
        },
        "performance":    compute_performance(pnl),
        "signal_quality": {},
        "stability":      compute_stability(pnl, weights_df),
        "attribution":    {},
        "diagnostics":    {},
        "regime":         None,
    }

    # IC per alpha (stored in signal_quality)
    for name, alpha_df in alpha_dict.items():
        print(f"  Computing IC: {name} ...", end=" ", flush=True)
        report["signal_quality"][name] = compute_ic(alpha_df, returns_df)
        print(f"mean_IC={report['signal_quality'][name].get('mean_ic')}")

    # Attribution
    if standalone:
        report["attribution"] = compute_attribution(standalone)

    # Diagnostics
    print("  Computing diagnostics ...", end=" ", flush=True)
    report["diagnostics"] = compute_diagnostics(weights_df, alpha_dict, returns_df)
    print("done")

    # Regime
    if regime_df is not None or scale_series is not None:
        report["regime"] = compute_regime_stats(regime_df, scale_series, dcc_converged)

    # Save
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    fname = out_path / f"report_{phase_name.lower().replace(' ', '_').replace('-', '_')}.json"

    with open(fname, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"  Saved: {fname}")
    return report
