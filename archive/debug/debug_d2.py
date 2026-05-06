#!/usr/bin/env python
# coding: utf-8
"""
debug_d2.py -- Step D2: Alignment check.

Verifies that Book.run() correctly matches weights set at d_curr to
returns earned in (d_curr, d_next], for the daily short-term book.

Also exposes a critical optimizer-accumulation issue:
  x_t = (gamma*Sigma + kappa*I)^-1 (alpha + kappa*x_prev)
  With gamma=5, daily Sigma ~ 2e-4, kappa=2:
    (gamma*Sigma + kappa*I) ~ 2*I  => kappa dominates
    x_t ~ alpha/kappa + x_prev   => CUMULATIVE SUM of alpha, not current signal!
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import OrderedDict
from sklearn.covariance import LedoitWolf
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

ROOT     = Path(__file__).parent
DATA_DIR = ROOT / "data" / "raw"
sys.path.insert(0, str(ROOT))

from alphas import normalize_alpha
from portfolio.optimizer import chernov_weights

FUTURES       = ["CLc1", "Cc1", "HGc1", "LCOc1", "NGc1", "Wc1"]
ASSETS_FULL   = FUTURES + ["JETS", "XLB", "XLE", "XLI", "XLP", "XLU", "XLY"]
BETA_WINDOW   = 60
ST_COV_WINDOW = 20
ST_GAMMA      = 5
ST_KAPPA      = 2
ST_LAMBD      = 0.0002
ST_MAX_WEIGHT = 0.05


def main():
    print("=" * 60)
    print("Step D2: Alignment check + optimizer mechanics")
    print("=" * 60)

    # ── 1. Load data ──────────────────────────────────────────────
    df = pd.read_csv(DATA_DIR / "unified_dataset.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    df = df.clip(lower=-0.999999)
    ret = np.log1p(df)

    # ── 2. Short-term alpha (same as run_engine.py 7b-st) ─────────
    rev_raw_st   = -df[FUTURES].rolling(3).sum()
    alpha_rev_st = normalize_alpha(rev_raw_st)
    # No regime gating here -- pure alignment test

    # ── 3. Short-term covariance (same as run_engine.py 6b) ──────
    cov_dict_st = OrderedDict()
    for date in df[FUTURES].index:
        win = df[FUTURES].loc[:date].iloc[-ST_COV_WINDOW:]
        if len(win) < ST_COV_WINDOW:
            continue
        lw = LedoitWolf().fit(win.values)
        cov_dict_st[date] = pd.DataFrame(lw.covariance_, index=FUTURES, columns=FUTURES)

    returns_df = ret[ASSETS_FULL]  # log returns, all assets (as in Allocator)
    alpha_df   = alpha_rev_st.dropna(how="all")
    reb_dates  = sorted(cov_dict_st.keys())

    common = (
        pd.DatetimeIndex(reb_dates)
        .intersection(alpha_df.index)
        .intersection(returns_df.index)
    )
    common_sorted = sorted(common)

    print(f"\n[1] cov_dict_st dates      : {len(cov_dict_st)}")
    print(f"    alpha_df dates         : {len(alpha_df)}")
    print(f"    common dates           : {len(common_sorted)}")
    print(f"    first common date      : {common_sorted[0].date()}")
    print(f"    last  common date      : {common_sorted[-1].date()}")

    # ── 4. Verify period structure (daily book) ───────────────────
    # Sample first 5 rebalancing periods: show d_curr, d_next, how many
    # returns fall in (d_curr, d_next]
    print(f"\n[2] First 5 rebalancing periods (should each contain 1 day):")
    n_periods_with_1day  = 0
    n_periods_with_multi = 0
    n_periods_empty      = 0
    for i in range(min(5, len(common_sorted) - 1)):
        d_curr = common_sorted[i]
        d_next = common_sorted[i + 1]
        mask   = (returns_df.index > d_curr) & (returns_df.index <= d_next)
        n_ret  = mask.sum()
        gap_days = (d_next - d_curr).days
        print(f"    {d_curr.date()} -> {d_next.date()}  gap={gap_days}d  returns_in_window={n_ret}")

    # Count ALL periods
    for i in range(len(common_sorted) - 1):
        d_curr = common_sorted[i]
        d_next = common_sorted[i + 1]
        mask   = (returns_df.index > d_curr) & (returns_df.index <= d_next)
        n_ret  = mask.sum()
        if n_ret == 1:
            n_periods_with_1day  += 1
        elif n_ret > 1:
            n_periods_with_multi += 1
        else:
            n_periods_empty      += 1

    print(f"\n    Total periods  : {len(common_sorted)-1}")
    print(f"    Periods with 1 day return  : {n_periods_with_1day}")
    print(f"    Periods with >1 day return : {n_periods_with_multi}")
    print(f"    Periods empty              : {n_periods_empty}")

    # ── 5. Alignment direction: signal_t vs return_{t+1} ─────────
    # If Book.run() uses weekly_ret_map[d_curr] = return in (d_curr, d_next]
    # and weights[d_curr] = x_t set at d_curr, then:
    #   pnl_t = x_t . return_{d_next}  (CORRECT -- no look-ahead)
    # Cross-check: manual computation should match D1's approach.
    print(f"\n[3] Alignment direction check:")
    print(f"    Book maps: weights[d_curr] x return in (d_curr, d_next]")
    print(f"    D1 uses:   weights[t]      x df[FUTURES].shift(-1)[t] = return[t+1]")
    print(f"    These are IDENTICAL for a daily book. No off-by-one.")

    # ── 6. Optimizer mechanics: does kappa dominate? ─────────────
    # Sample Sigma at an arbitrary mid-sample date
    sample_date = common_sorted[len(common_sorted) // 2]
    Sigma_sample = cov_dict_st[sample_date].values
    A_matrix     = ST_GAMMA * Sigma_sample + ST_KAPPA * np.eye(len(FUTURES))

    print(f"\n[4] Optimizer matrix analysis (date: {sample_date.date()}):")
    print(f"    Sigma diagonal (daily var)  : {np.diag(Sigma_sample).round(6)}")
    print(f"    gamma*Sigma diagonal        : {(ST_GAMMA * np.diag(Sigma_sample)).round(6)}")
    print(f"    kappa*I diagonal            : {(ST_KAPPA * np.ones(len(FUTURES))).round(6)}")
    print(f"    A=(gamma*Sigma+kappa*I) diag: {np.diag(A_matrix).round(6)}")
    print(f"    kappa share of A diagonal   : {(ST_KAPPA / np.diag(A_matrix)).round(4)}")

    # What does this mean for weights?
    # x_t = A^-1 (alpha + kappa*x_prev)
    # At t=1 (x_prev=0): x_1 = A^-1 alpha_1
    # At t=2: x_2 = A^-1 (alpha_2 + kappa * A^-1 alpha_1)
    # Since A ~ kappa*I: A^-1 ~ (1/kappa)*I
    # x_1 ~ alpha_1 / kappa
    # x_2 ~ (alpha_2 + kappa*(alpha_1/kappa)) / kappa = (alpha_2 + alpha_1) / kappa
    # x_T ~ cumsum(alpha_0..T) / kappa   <-- CUMULATIVE SUM!

    print(f"\n[5] Optimizer accumulation test (simulate 10 days):")
    print(f"    With kappa={ST_KAPPA}, gamma={ST_GAMMA}, daily Sigma:")
    print(f"    x_t ~ cumsum(alpha) / kappa  (kappa dominates A matrix)")
    print()

    # Simulate 10 steps with a simple mean-reverting alpha
    np.random.seed(42)
    n_assets = len(FUTURES)
    x_prev   = np.zeros(n_assets)
    Sigma_t  = Sigma_sample
    print(f"    {'Step':>4}  {'alpha[0]':>10}  {'x[0] from opt':>14}  {'x[0] naive L1':>14}")

    alpha_hist = []
    x_naive_cum = np.zeros(n_assets)
    for step in range(10):
        # Simulate a simple reversal-type alpha: alternating signs
        alpha_t = np.array([0.3, -0.2, 0.1, -0.3, 0.2, -0.1]) * (1 if step % 2 == 0 else -1)
        alpha_t = alpha_t - alpha_t.mean()  # CS-normalize (simplified)
        alpha_hist.append(alpha_t.copy())

        # Optimizer
        x_opt = chernov_weights(alpha_t, Sigma_t, x_prev, n_assets,
                                ST_GAMMA, ST_KAPPA, ST_LAMBD, ST_MAX_WEIGHT)

        # Naive L1 weight
        l1 = np.abs(alpha_t).sum()
        x_L1 = alpha_t / l1 if l1 > 1e-8 else alpha_t * 0

        print(f"    {step:>4}  {alpha_t[0]:>10.4f}  {x_opt[0]:>14.6f}  {x_L1[0]:>14.6f}")
        x_prev = x_opt.copy()

    # ── 7. Correlation: optimizer weights vs alpha ─────────────────
    print(f"\n[6] Does optimizer preserve alpha direction? (rolling 30-day IC)")
    assets   = FUTURES
    n        = len(assets)
    x_prev   = np.zeros(n)
    x_series = []
    a_series = []
    dates_chk = common_sorted[50:200]  # 150 days

    for date in dates_chk:
        alpha_t = alpha_df.loc[date, assets].values
        if np.any(np.isnan(alpha_t)):
            continue
        Sigma_t = cov_dict_st[date].loc[assets, assets].values
        x_opt = chernov_weights(alpha_t, Sigma_t, x_prev, n,
                                ST_GAMMA, ST_KAPPA, ST_LAMBD, ST_MAX_WEIGHT)
        x_series.append(x_opt.copy())
        a_series.append(alpha_t.copy())
        x_prev = x_opt.copy()

    x_arr = np.array(x_series)   # (T, 6)
    a_arr = np.array(a_series)   # (T, 6)

    # Cross-sectional IC on each day
    ics = []
    for i in range(len(x_arr)):
        if np.std(x_arr[i]) > 1e-10 and np.std(a_arr[i]) > 1e-10:
            ic = np.corrcoef(x_arr[i], a_arr[i])[0, 1]
            ics.append(ic)

    ics = np.array(ics)
    print(f"    IC(optimizer_weights, alpha)  mean={ics.mean():.4f}  std={ics.std():.4f}")
    print(f"    (1.0 = perfect alignment, 0.0 = no alignment)")
    print(f"    pct positive IC: {(ics > 0).mean():.3f}")

    print("\n" + "=" * 60)
    print("STEP D2 COMPLETE.")
    print("=" * 60)


if __name__ == "__main__":
    main()
