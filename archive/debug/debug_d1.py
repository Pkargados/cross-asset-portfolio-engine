#!/usr/bin/env python
# coding: utf-8
"""
debug_d1.py — Step D1: Pure signal sanity check (outside engine).

Constructs the reversal signal directly on raw simple returns (FUTURES only),
builds naive L1-normalized weights, and computes daily PnL and Sharpe.
No optimizer, no Book, no regime gating — pure signal test.

Expected: positive daily Sharpe, consistent with signal_research.py findings.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT     = Path(__file__).parent
DATA_DIR = ROOT / "data" / "raw"
sys.path.insert(0, str(ROOT))

from alphas import normalize_alpha   # CS z-score + MAD winsorize

FUTURES = ["CLc1", "Cc1", "HGc1", "LCOc1", "NGc1", "Wc1"]

def main():
    print("=" * 60)
    print("Step D1: Pure signal sanity check (outside engine)")
    print("=" * 60)

    # ── 1. Load data ──────────────────────────────────────────────
    df = pd.read_csv(DATA_DIR / "unified_dataset.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    df = df.clip(lower=-0.999999)

    print(f"\n[1] Data loaded: {df.shape}  ({df.index[0].date()} to {df.index[-1].date()})")

    # ── 2. Build alpha (identical to run_engine.py section 7b-st) ─
    # rev_raw_st = -df[FUTURES].rolling(3).sum()  (raw simple returns, window=3)
    rev_raw_st   = -df[FUTURES].rolling(3).sum()
    alpha_rev_st = normalize_alpha(rev_raw_st)

    print(f"\n[2] alpha_rev_st shape      : {alpha_rev_st.dropna(how='all').shape}")
    print(f"    alpha_rev_st mean|col    : {alpha_rev_st.mean().to_dict()}")

    # ── 3. Naive L1-normalized weights ────────────────────────────
    # weights_t = alpha_t / sum(|alpha_t|)  — unit gross exposure
    l1_sum   = alpha_rev_st.abs().sum(axis=1).replace(0, np.nan)
    weights  = alpha_rev_st.div(l1_sum, axis=0)

    print(f"\n[3] Weights gross exposure   : mean={weights.abs().sum(axis=1).mean():.4f}"
          f"  (should be ~1.0 where non-NaN)")

    # ── 4. PnL: weights_t × return_{t+1} ─────────────────────────
    # shift(-1) aligns signal at t with next-day return
    next_ret = df[FUTURES].shift(-1)
    pnl_daily = (weights * next_ret).sum(axis=1).dropna()

    print(f"\n[4] PnL observations         : {len(pnl_daily)}")
    print(f"    PnL mean (daily)         : {pnl_daily.mean():.6f}")
    print(f"    PnL std  (daily)         : {pnl_daily.std():.6f}")

    sharpe_daily = np.sqrt(252) * pnl_daily.mean() / pnl_daily.std()
    print(f"\n    Daily Sharpe (sqrt252)   : {sharpe_daily:.4f}  (should be positive)")

    # ── 5. Split pre/post 2020 ────────────────────────────────────
    pre  = pnl_daily[pnl_daily.index <  "2020-01-01"]
    post = pnl_daily[pnl_daily.index >= "2020-01-01"]

    def sharpe(s):
        return np.sqrt(252) * s.mean() / s.std() if s.std() > 1e-12 else np.nan

    print(f"\n[5] Pre-2020  Sharpe          : {sharpe(pre):.4f}  (n={len(pre)})")
    print(f"    Post-2020 Sharpe          : {sharpe(post):.4f}  (n={len(post)})")

    # ── 6. Per-asset IC ───────────────────────────────────────────
    print(f"\n[6] Per-asset IC (signal vs next-day return):")
    for col in FUTURES:
        sig = alpha_rev_st[col].dropna()
        ret = df[col].shift(-1).reindex(sig.index).dropna()
        idx = sig.index.intersection(ret.index)
        ic  = sig.loc[idx].corr(ret.loc[idx])
        print(f"    {col:8s}  IC = {ic:.4f}")

    # ── 7. Window sensitivity: try windows 2, 3, 5 ──────────────
    print(f"\n[7] Window sensitivity (L1-norm weights, daily Sharpe):")
    for w in [2, 3, 5]:
        raw   = -df[FUTURES].rolling(w).sum()
        alpha = normalize_alpha(raw)
        l1    = alpha.abs().sum(axis=1).replace(0, np.nan)
        wt    = alpha.div(l1, axis=0)
        pnl   = (wt * df[FUTURES].shift(-1)).sum(axis=1).dropna()
        sr    = np.sqrt(252) * pnl.mean() / pnl.std() if pnl.std() > 1e-12 else np.nan
        print(f"    window={w}  Sharpe={sr:.4f}  n={len(pnl)}")

    print("\n" + "=" * 60)
    print("STEP D1 COMPLETE.")
    print("=" * 60)


if __name__ == "__main__":
    main()
