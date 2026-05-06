#!/usr/bin/env python
# coding: utf-8
"""
debug_d4.py -- Steps D3 (lightweight) + D4: Isolate which component
               breaks the reversal signal inside Book.run().

No DCC needed. Uses synthetic regime (scale=0.5 everywhere = "all normal")
to test each layer of the engine pipeline.

Scenarios (each builds on the previous):
  A  D1 baseline:      ungated, L1-norm,   simple returns       -> expected ~+0.073
  B  Gated only:       gated*0.5, L1-norm, simple returns       -> L1 re-norms, same as A
  C  + Optimizer:      gated*0.5, chernov, simple returns, no vol-scale
  D  + Log returns:    gated*0.5, chernov, log    returns, no vol-scale
  E  + Vol scaling:    gated*0.5, chernov, log    returns, + EWMA vol-scale
  F  Book.run() proxy: gated*0.5 alpha fed into Book.run()
  G  Book.run() ungated: raw alpha fed into Book.run()

If any step flips sign, that component is the bug.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import OrderedDict
from sklearn.covariance import LedoitWolf

ROOT     = Path(__file__).parent
DATA_DIR = ROOT / "data" / "raw"
sys.path.insert(0, str(ROOT))

from alphas import normalize_alpha
from portfolio.optimizer import chernov_weights
from portfolio.book import Book

FUTURES       = ["CLc1", "Cc1", "HGc1", "LCOc1", "NGc1", "Wc1"]
ASSETS_FULL   = FUTURES + ["JETS", "XLB", "XLE", "XLI", "XLP", "XLU", "XLY"]
ST_COV_WINDOW = 20
ST_GAMMA      = 5
ST_KAPPA      = 2
ST_LAMBD      = 0.0002
ST_MAX_WEIGHT = 0.05
ST_TARGET_VOL = 0.10
ST_EWMA_HL    = 1
SCALE_MIN     = 0.2
SCALE_MAX     = 1.5


def build_cov(df_futures, window=20):
    cov_dict = OrderedDict()
    for date in df_futures.index:
        win = df_futures.loc[:date].iloc[-window:]
        if len(win) < window:
            continue
        lw = LedoitWolf().fit(win.values)
        cov_dict[date] = pd.DataFrame(lw.covariance_, index=FUTURES, columns=FUTURES)
    return cov_dict


def run_manual(alpha_df, cov_dict, returns_df, use_vol_scaling=False):
    """
    Manual backtest: optimizer + optional vol-scaling, no Book.run() wrapper.
    Returns pd.Series of per-period PnL.
    """
    assets = FUTURES
    n      = len(assets)
    common = sorted(
        pd.DatetimeIndex(list(cov_dict.keys()))
        .intersection(alpha_df.dropna(how="all").index)
        .intersection(returns_df.index)
    )

    # Pre-build next-period return map
    ret_map = {}
    for i in range(len(common) - 1):
        d_curr, d_next = common[i], common[i + 1]
        mask = (returns_df.index > d_curr) & (returns_df.index <= d_next)
        wret = returns_df.loc[mask, assets]
        if len(wret) > 0:
            ret_map[d_curr] = wret.values.sum(axis=0)

    ewma_a  = 1.0 - np.exp(-np.log(2.0) / ST_EWMA_HL)
    ewma_v  = (ST_TARGET_VOL / np.sqrt(52)) ** 2
    x_prev  = np.zeros(n)
    prev_x  = None
    prev_dt = None
    pnl_list = []
    dt_list  = []

    for date in common:
        # EWMA update
        if prev_x is not None and prev_dt in ret_map:
            pnl_t = float(np.dot(prev_x, ret_map[prev_dt]))
            ewma_v = (1 - ewma_a) * ewma_v + ewma_a * pnl_t ** 2

        alpha_t = alpha_df.loc[date, assets].values
        if np.any(np.isnan(alpha_t)):
            x_prev = np.zeros(n)
            continue

        Sigma_t = cov_dict[date].loc[assets, assets].values
        x_t = chernov_weights(alpha_t, Sigma_t, x_prev, n,
                              ST_GAMMA, ST_KAPPA, ST_LAMBD, ST_MAX_WEIGHT)

        if use_vol_scaling:
            rv = float(np.sqrt(max(ewma_v, 0.0) * 52))
            if rv > 1e-8:
                scale_raw = (ST_TARGET_VOL / rv) ** 2.0
                max_abs   = float(np.max(np.abs(x_t)))
                if max_abs > 1e-10:
                    scale_raw = min(scale_raw, ST_MAX_WEIGHT / max_abs)
                scale = float(np.clip(scale_raw, SCALE_MIN, SCALE_MAX))
                x_t   = x_t * scale

        x_t = x_t - x_t.mean()
        x_t = np.clip(x_t, -ST_MAX_WEIGHT, ST_MAX_WEIGHT)

        if date in ret_map:
            pnl_list.append(float(np.dot(x_t, ret_map[date])))
            dt_list.append(date)

        x_prev  = x_t.copy()
        prev_x  = x_t.copy()
        prev_dt = date

    return pd.Series(pnl_list, index=dt_list)


def sharpe(s, freq=252):
    s = s.dropna()
    return np.sqrt(freq) * s.mean() / s.std() if s.std() > 1e-12 else np.nan


def main():
    print("=" * 60)
    print("Step D4: Isolate engine bug (no DCC needed)")
    print("=" * 60)

    # ── 1. Load data ──────────────────────────────────────────────
    df = pd.read_csv(DATA_DIR / "unified_dataset.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    df = df.clip(lower=-0.999999)
    ret = np.log1p(df)
    print(f"\n[1] Data: {df.shape}")

    # ── 2. Build alpha ────────────────────────────────────────────
    rev_raw_st   = -df[FUTURES].rolling(3).sum()
    alpha_rev_st = normalize_alpha(rev_raw_st)
    # Synthetic "all normal" regime: scale=0.5 everywhere
    alpha_gated  = alpha_rev_st * 0.5

    # ── 3. Build covariance ───────────────────────────────────────
    print("[2] Building short-term covariance (window=20) ...")
    cov_dict_st = build_cov(df[FUTURES], window=ST_COV_WINDOW)
    print(f"    cov_dict_st: {len(cov_dict_st)} dates")

    # ── SCENARIO A: D1 baseline ───────────────────────────────────
    l1 = alpha_rev_st.abs().sum(axis=1).replace(0, np.nan)
    pnl_A = (alpha_rev_st.div(l1, axis=0) * df[FUTURES].shift(-1)).sum(axis=1).dropna()

    # ── SCENARIO B: gated*0.5, L1-norm, simple returns ───────────
    # L1-norm re-normalizes, so scale factor cancels -> same as A
    l1g = alpha_gated.abs().sum(axis=1).replace(0, np.nan)
    pnl_B = (alpha_gated.div(l1g, axis=0) * df[FUTURES].shift(-1)).sum(axis=1).dropna()

    # ── SCENARIO C: gated*0.5, optimizer, simple returns, no vol-scale ─
    print("[3] Scenario C (optimizer, no vol-scale, simple returns) ...")
    pnl_C = run_manual(alpha_gated, cov_dict_st, df[ASSETS_FULL], use_vol_scaling=False)

    # ── SCENARIO D: gated*0.5, optimizer, log returns, no vol-scale ────
    print("[4] Scenario D (optimizer, no vol-scale, log returns) ...")
    pnl_D = run_manual(alpha_gated, cov_dict_st, ret[ASSETS_FULL], use_vol_scaling=False)

    # ── SCENARIO E: gated*0.5, optimizer, log returns, + vol scale ─────
    print("[5] Scenario E (optimizer, vol-scale, log returns) ...")
    pnl_E = run_manual(alpha_gated, cov_dict_st, ret[ASSETS_FULL], use_vol_scaling=True)

    # ── SCENARIO F: Book.run() with gated alpha ───────────────────
    print("[6] Scenario F (Book.run, gated alpha) ...")
    book_f = Book(
        name="st_proxy", alpha_df=alpha_gated,
        cov_dict=cov_dict_st, reb_dates=list(cov_dict_st.keys()),
        gamma=ST_GAMMA, kappa=ST_KAPPA, lambd=ST_LAMBD,
        max_weight=ST_MAX_WEIGHT, target_vol=ST_TARGET_VOL,
        ewma_halflife=ST_EWMA_HL, scale_min=SCALE_MIN, scale_max=SCALE_MAX,
    )
    res_F = book_f.run(ret[ASSETS_FULL])
    pnl_F = res_F["pnl"]

    # ── SCENARIO G: Book.run() with ungated alpha ─────────────────
    print("[7] Scenario G (Book.run, ungated alpha) ...")
    book_g = Book(
        name="st_ungated", alpha_df=alpha_rev_st,
        cov_dict=cov_dict_st, reb_dates=list(cov_dict_st.keys()),
        gamma=ST_GAMMA, kappa=ST_KAPPA, lambd=ST_LAMBD,
        max_weight=ST_MAX_WEIGHT, target_vol=ST_TARGET_VOL,
        ewma_halflife=ST_EWMA_HL, scale_min=SCALE_MIN, scale_max=SCALE_MAX,
    )
    res_G = book_g.run(ret[ASSETS_FULL])
    pnl_G = res_G["pnl"]

    # ── Results table ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    header = f"{'Scenario':<52}  {'Sharpe(252)':>11}  {'MeanPnL':>10}  {'n':>5}"
    print(header)
    print("-" * 82)

    rows = [
        ("A  D1 baseline (ungated, L1, simple)",       pnl_A),
        ("B  gated*0.5, L1-norm (scale cancels=A?)",   pnl_B),
        ("C  + Optimizer, simple, no vol-scale",       pnl_C),
        ("D  + Log returns, no vol-scale",             pnl_D),
        ("E  + Vol scaling (EWMA halflife=1)",         pnl_E),
        ("F  Book.run() gated*0.5 alpha",              pnl_F),
        ("G  Book.run() ungated alpha",                pnl_G),
    ]
    for name, pnl in rows:
        pnl = pnl.dropna()
        sr  = sharpe(pnl)
        mn  = pnl.mean()
        print(f"{name:<52}  {sr:>11.4f}  {mn:>10.6f}  {len(pnl):>5}")

    # ── Detailed diagnostics: Book.run() G ───────────────────────
    print(f"\n[8] Book.run() G (ungated) internal diagnostics:")
    pnl_G_s = res_G["pnl"].dropna()
    w_G = res_G["weights"]
    print(f"    avg_scale    : {res_G['avg_scale']}")
    print(f"    n_cap_bind   : {res_G['n_cap_bind']}")
    print(f"    gross_exp    : {w_G.abs().sum(axis=1).mean():.4f}")
    print(f"    net_exp mean : {w_G.sum(axis=1).mean():.6f}")
    print(f"    reported sharpe (sqrt52) : {res_G['sharpe']}")
    print(f"    correct sharpe (sqrt252) : {sharpe(pnl_G_s):.4f}")

    # ── EWMA scale trajectory with realistic daily PnL ────────────
    print(f"\n[9] EWMA vol-scale trajectory (realistic daily PnL ~ 0.001):")
    ewma_a  = 1.0 - np.exp(-np.log(2.0) / ST_EWMA_HL)
    ewma_v0 = (ST_TARGET_VOL / np.sqrt(52)) ** 2
    ewma_v  = ewma_v0
    print(f"    init: ewma_var={ewma_v:.2e}  rv={np.sqrt(ewma_v*52):.4f}  scale=1.0")
    for step in range(15):
        pnl_t  = 0.001
        ewma_v = (1 - ewma_a) * ewma_v + ewma_a * pnl_t**2
        rv     = np.sqrt(ewma_v * 52)
        scale  = np.clip((ST_TARGET_VOL / rv)**2, SCALE_MIN, SCALE_MAX)
        print(f"    step={step+1:2d}: ewma_var={ewma_v:.2e}  rv={rv:.4f}  scale={scale:.4f}")

    print("\n" + "=" * 60)
    print("STEP D4 COMPLETE.")
    print("=" * 60)


if __name__ == "__main__":
    main()
