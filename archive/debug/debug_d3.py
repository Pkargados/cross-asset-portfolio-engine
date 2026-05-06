#!/usr/bin/env python
# coding: utf-8
"""
debug_d3.py -- Step D3: Regime gating validation + alpha scale analysis.

Checks:
  1. Regime label distribution from regime_df["regime"].value_counts()
  2. rev_scale_series distribution (how often active, at what scale)
  3. Does alpha_rev_gated still have predictive power (IC vs next-day return)?
  4. Comparison: engine alpha (gated) vs D1 alpha (ungated) IC
  5. Direct PnL with gated alpha but L1-norm weights (no optimizer)
     to isolate regime gating effect from optimizer effect.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

ROOT     = Path(__file__).parent
DATA_DIR = ROOT / "data" / "raw"
sys.path.insert(0, str(ROOT))

from alphas import normalize_alpha
from regime.regime_detection import compute_regime_signals
from alphas.spread import build_spread

FUTURES     = ["CLc1", "Cc1", "HGc1", "LCOc1", "NGc1", "Wc1"]
ETFS        = ["JETS", "XLB", "XLE", "XLI", "XLP", "XLU", "XLY"]
ASSETS      = FUTURES + ETFS
BETA_WINDOW = 60
PAIRS = [
    ("XLE",  "CLc1"), ("XLE",  "LCOc1"), ("XLE",  "NGc1"),
    ("XLI",  "HGc1"), ("XLB",  "HGc1"),  ("XLB",  "Cc1"),
    ("XLB",  "Wc1"),  ("XLY",  "Wc1"),   ("JETS", "CLc1"),
]
REV_REGIME_SCALE = {"clustered": 1.0, "normal": 0.5, "crowded": 0.0, "crisis": 0.0}


def main():
    print("=" * 60)
    print("Step D3: Regime gating validation")
    print("=" * 60)

    # ── 1. Load data ──────────────────────────────────────────────
    df = pd.read_csv(DATA_DIR / "unified_dataset.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    df = df.clip(lower=-0.999999)
    ret = np.log1p(df)
    print(f"\n[1] Data: {df.shape}  ({df.index[0].date()} to {df.index[-1].date()})")

    # ── 2. Beta neutralization (needed for regime detection) ──────
    print("\n[2] Beta neutralization ...")
    mkt          = ret["SPX"]
    residual_ret = pd.DataFrame(index=ret.index, columns=ASSETS, dtype=float)
    for asset in ASSETS:
        y    = ret[asset]
        X    = sm.add_constant(mkt)
        rols = RollingOLS(y, X, window=BETA_WINDOW).fit()
        residual_ret[asset] = y - (rols.params["const"] + rols.params["SPX"] * mkt)
    residual_ret = residual_ret.dropna()
    print(f"    Residual returns: {residual_ret.shape}")

    # ── 3. Spread raw (needed for regime detection) ───────────────
    print("\n[3] Building spread (for regime detection input) ...")
    _, spread_raw = build_spread(
        residual_ret, pairs=PAIRS, window=120, adf_pval=0.10,
        halflife_max=30, verbose=False,
    )

    # ── 4. Regime detection ───────────────────────────────────────
    print("\n[4] Computing regime signals ...")
    regime_df = compute_regime_signals(residual_ret[ASSETS], spread_raw)

    print(f"\n    regime_df columns  : {list(regime_df.columns)}")
    print(f"    regime_df shape    : {regime_df.shape}")

    # Regime label distribution
    regime_counts = regime_df["regime"].value_counts()
    total         = len(regime_df)
    print(f"\n    Regime label distribution:")
    for lbl, cnt in regime_counts.items():
        pct = cnt / total * 100
        print(f"      {lbl:15s}  {cnt:5d}  ({pct:.1f}%)")

    # ── 5. Build alpha + regime scale ─────────────────────────────
    print("\n[5] Building reversal alpha and applying regime gate ...")
    rev_raw_st   = -df[FUTURES].rolling(3).sum()
    alpha_rev_st = normalize_alpha(rev_raw_st)

    regime_sorted = regime_df.sort_index()
    scale_rows    = {}
    label_rows    = {}
    for date in alpha_rev_st.index:
        avail = regime_sorted.loc[:date]
        if len(avail) == 0:
            lbl = "normal"
        else:
            lbl = avail["regime"].iloc[-1]
            if not isinstance(lbl, str) or pd.isna(lbl):
                lbl = "normal"
        label_rows[date] = lbl
        scale_rows[date] = REV_REGIME_SCALE.get(lbl, 0.5)

    rev_scale_series  = pd.Series(scale_rows)
    rev_label_series  = pd.Series(label_rows)
    alpha_rev_gated   = alpha_rev_st.mul(rev_scale_series, axis=0)

    # Scale distribution
    scale_counts = rev_scale_series.value_counts()
    print(f"\n    rev_scale_series distribution:")
    for scale, cnt in sorted(scale_counts.items()):
        pct     = cnt / len(rev_scale_series) * 100
        gate    = {0.0: "KILLED", 0.5: "half", 1.0: "FULL"}
        label_for_scale = {
            0.0: "crowded/crisis",
            0.5: "normal",
            1.0: "clustered",
        }
        print(f"      scale={scale}  {cnt:5d}  ({pct:.1f}%)  [{label_for_scale.get(scale, '?')}]")

    n_active = int((rev_scale_series > 0).sum())
    n_killed = int((rev_scale_series == 0).sum())
    print(f"\n    Active (scale>0)  : {n_active}/{len(rev_scale_series)} ({n_active/len(rev_scale_series)*100:.1f}%)")
    print(f"    Killed (scale==0) : {n_killed}/{len(rev_scale_series)} ({n_killed/len(rev_scale_series)*100:.1f}%)")

    # ── 6. IC: gated alpha vs next-day return ─────────────────────
    print(f"\n[6] IC: alpha vs next-day return (ungated vs gated):")
    next_ret_fut = df[FUTURES].shift(-1)

    def mean_ic(alpha_df, next_ret_df):
        ics = []
        common_idx = alpha_df.dropna(how="all").index.intersection(next_ret_df.dropna(how="all").index)
        for date in common_idx:
            a = alpha_df.loc[date].dropna()
            r = next_ret_df.loc[date].reindex(a.index).dropna()
            idx = a.index.intersection(r.index)
            if len(idx) >= 3 and a.loc[idx].std() > 1e-10 and r.loc[idx].std() > 1e-10:
                ics.append(a.loc[idx].corr(r.loc[idx]))
        ics = np.array(ics)
        return ics.mean() if len(ics) > 0 else np.nan, ics.std() if len(ics) > 0 else np.nan, len(ics)

    ic_ungated_mean, ic_ungated_std, n1 = mean_ic(alpha_rev_st, next_ret_fut)
    ic_gated_mean,   ic_gated_std,   n2 = mean_ic(alpha_rev_gated, next_ret_fut)

    print(f"    Ungated alpha IC  : mean={ic_ungated_mean:.4f}  std={ic_ungated_std:.4f}  n={n1}")
    print(f"    Gated alpha IC    : mean={ic_gated_mean:.4f}  std={ic_gated_std:.4f}  n={n2}")

    # ── 7. Direct PnL (L1-norm, no optimizer) ────────────────────
    # Test regime-gated alpha with L1-norm weights only (no optimizer, no vol scaling)
    # to isolate regime gating effect from optimizer/vol-scaling effects
    print(f"\n[7] Direct L1-norm PnL with regime-gated alpha:")
    l1_sum_gated = alpha_rev_gated.abs().sum(axis=1).replace(0, np.nan)
    w_gated      = alpha_rev_gated.div(l1_sum_gated, axis=0)

    # PnL = weights * next-day return (same alignment as D1)
    pnl_gated = (w_gated * df[FUTURES].shift(-1)).sum(axis=1).dropna()

    # Compare with ungated (D1 result)
    l1_sum_ungated = alpha_rev_st.abs().sum(axis=1).replace(0, np.nan)
    w_ungated      = alpha_rev_st.div(l1_sum_ungated, axis=0)
    pnl_ungated    = (w_ungated * df[FUTURES].shift(-1)).sum(axis=1).dropna()

    def sharpe(s, freq=252):
        return np.sqrt(freq) * s.mean() / s.std() if s.std() > 1e-12 else np.nan

    print(f"    Ungated L1 Sharpe (D1)  : {sharpe(pnl_ungated):.4f}  (n={len(pnl_ungated)})")
    print(f"    Gated L1 Sharpe         : {sharpe(pnl_gated):.4f}  (n={len(pnl_gated)})")

    # ── 8. Active-only PnL (only dates when scale > 0) ───────────
    active_dates = rev_scale_series[rev_scale_series > 0].index
    pnl_active   = pnl_ungated.reindex(active_dates).dropna()
    pnl_killed   = pnl_ungated[~pnl_ungated.index.isin(active_dates)].dropna()

    print(f"\n[8] PnL breakdown: active vs killed dates:")
    print(f"    Active dates Sharpe     : {sharpe(pnl_active):.4f}  (n={len(pnl_active)})")
    print(f"    Killed dates Sharpe     : {sharpe(pnl_killed):.4f}  (n={len(pnl_killed)})")
    print(f"    (If killed dates have higher Sharpe, gating is hurting!)")

    # ── 9. Label-wise breakdown ───────────────────────────────────
    print(f"\n[9] PnL by regime label (ungated alpha, L1-norm weights):")
    for lbl in ["clustered", "normal", "crowded", "crisis"]:
        dates_lbl = rev_label_series[rev_label_series == lbl].index
        dates_lbl = dates_lbl.intersection(pnl_ungated.index)
        if len(dates_lbl) > 5:
            s = sharpe(pnl_ungated.loc[dates_lbl])
            print(f"    {lbl:15s}  n={len(dates_lbl):5d}  Sharpe={s:.4f}")

    print("\n" + "=" * 60)
    print("STEP D3 COMPLETE.")
    print("=" * 60)


if __name__ == "__main__":
    main()
