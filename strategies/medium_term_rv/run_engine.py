#!/usr/bin/env python
# coding: utf-8
"""
run_engine.py — Medium-term statistical arbitrage strategy (Momentum + Spread).

Runs the full pipeline using the modular engine (Book + Allocator).
Independent of run_baseline.py — no legacy backtest dependency.

Pipeline:
  1. Load data
  2. Beta neutralization (rolling OLS, window=60)
  3. Reversal, Momentum, Spread alpha construction
  4. FM alignment + re-normalization
  5. Rolling Sharpe combination — Mom + Spread only
  6. Ledoit-Wolf covariance (weekly, window=120)
  7. Engine-only execution: medium_term Book + short_term Book via Allocator

Usage:
    python strategies/medium_term_rv/run_engine.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import OrderedDict
from statsmodels.regression.rolling import RollingOLS
import statsmodels.api as sm
from sklearn.covariance import LedoitWolf

# ── path setup ──────────────────────────────────────────────────────────────
def _find_project_root() -> Path:
    for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]:
        if (p / "CLAUDE.md").exists():
            return p
    raise RuntimeError("Project root not found")

ROOT     = _find_project_root()
DATA_DIR = ROOT / "data" / "raw"
sys.path.insert(0, str(ROOT))

from engine.portfolio.book      import Book
from engine.portfolio.allocator import Allocator
from engine.alphas.momentum  import build_momentum
from engine.alphas.spread    import build_spread
from engine.alphas.reversal  import build_reversal
from engine.regime.regime_detection import compute_regime_signals
from engine.regime.regime_mapping   import get_book_actions

# ── parameters (IDENTICAL to run_baseline.py) ────────────────────────────────
BETA_WINDOW   = 60
REV_WINDOW    = 5
MOM_WINDOW    = 60
MOM_SKIP      = 5
SPREAD_WINDOW = 120
COV_WINDOW    = 120
ADF_PVAL      = 0.10
HALFLIFE_MAX  = 30
SHARPE_ROLL   = 52
SHRINK_NU     = 0.3
SMOOTH_HL     = 4
GAMMA         = 20
KAPPA         = 10
LAMBD         = 0.0002
MAX_WEIGHT    = 0.07
TARGET_VOL    = 0.15
EWMA_HALFLIFE = 2
SCALE_MIN     = 0.2
SCALE_MAX     = 1.5
ALIGN_WINDOW  = 52

# ── short-term book parameters ────────────────────────────────────────────────
ST_COV_WINDOW = 20
ST_GAMMA      = 5
ST_KAPPA      = 2
ST_MAX_WEIGHT = 0.05
ST_TARGET_VOL = 0.10
ST_EWMA_HL    = 1

FUTURES = ["CLc1", "Cc1", "HGc1", "LCOc1", "NGc1", "Wc1"]
ETFS    = ["JETS", "XLB", "XLE", "XLI", "XLP", "XLU", "XLY"]
ASSETS  = FUTURES + ETFS

PAIRS = [
    ("XLE",  "CLc1"),
    ("XLE",  "LCOc1"),
    ("XLE",  "NGc1"),
    ("XLI",  "HGc1"),
    ("XLB",  "HGc1"),
    ("XLB",  "Cc1"),
    ("XLB",  "Wc1"),
    ("XLY",  "Wc1"),
    ("JETS", "CLc1"),
]


# ============================================================================
# HELPERS (verbatim from run_baseline.py)
# ============================================================================

def cs_normalize(df: pd.DataFrame) -> pd.DataFrame:
    mu  = df.mean(axis=1, skipna=True)
    std = df.std(axis=1, skipna=True).clip(lower=1e-8)
    return df.sub(mu, axis=0).div(std, axis=0)


def robust_winsorize(df: pd.DataFrame, z: float = 3.0) -> pd.DataFrame:
    med = df.median(axis=1)
    mad = df.sub(med, axis=0).abs().median(axis=1).clip(lower=1e-8)
    return df.clip(lower=med - z * mad, upper=med + z * mad, axis=0)


def normalize_alpha(raw: pd.DataFrame) -> pd.DataFrame:
    return robust_winsorize(cs_normalize(raw))


def align_signal_to_expected_return(
    signal_daily: pd.DataFrame,
    returns_daily: pd.DataFrame,
    reb_dates,
    window: int = ALIGN_WINDOW,
    beta_floor: float = 0.0,
) -> tuple:
    reb_sorted = sorted(reb_dates)
    assets = [a for a in signal_daily.columns if a in returns_daily.columns]

    fwd_rows = {}
    for i in range(len(reb_sorted) - 1):
        d_curr, d_next = reb_sorted[i], reb_sorted[i + 1]
        mask = (returns_daily.index > d_curr) & (returns_daily.index <= d_next)
        wret = returns_daily.loc[mask, assets]
        if len(wret) > 0:
            fwd_rows[d_curr] = wret.sum(axis=0)

    fwd_df = pd.DataFrame(fwd_rows).T
    sig_weekly = signal_daily[assets].reindex(fwd_df.index, method="ffill")

    s_mat = sig_weekly.values
    r_mat = fwd_df.values
    T, N  = s_mat.shape
    alpha_vals = np.full((T, N), np.nan)
    raw_betas  = []

    for i in range(window, T):
        betas_fm = []
        for tau in range(i - window, i):
            sig_cs = s_mat[tau, :]
            ret_cs = r_mat[tau, :]
            valid  = ~(np.isnan(sig_cs) | np.isnan(ret_cs))
            if valid.sum() < 5:
                continue
            X_cs = np.column_stack([np.ones(valid.sum()), sig_cs[valid]])
            try:
                coef, _, _, _ = np.linalg.lstsq(X_cs, ret_cs[valid], rcond=None)
                betas_fm.append(coef[1])
            except Exception:
                pass

        if len(betas_fm) >= 10:
            raw_beta = float(np.mean(betas_fm))
            raw_betas.append(raw_beta)
            beta_t = float(max(beta_floor, raw_beta))
            alpha_vals[i, :] = beta_t * s_mat[i, :]

    alpha_df = pd.DataFrame(alpha_vals, index=fwd_df.index, columns=assets)
    for col in signal_daily.columns:
        if col not in alpha_df.columns:
            alpha_df[col] = np.nan

    rb = np.array(raw_betas)
    if len(rb) > 0:
        pct_neg     = float((rb < 0).mean())
        pct_clamped = float((rb < beta_floor).mean()) if np.isfinite(beta_floor) else 0.0
        beta_stats  = {
            "raw_mean":    round(float(rb.mean()),  6),
            "raw_std":     round(float(rb.std()),   6),
            "pct_negative": round(pct_neg,          4),
            "pct_clamped":  round(pct_clamped,      4),
            "n_estimates":  len(rb),
        }
    else:
        beta_stats = {"raw_mean": None, "raw_std": None,
                      "pct_negative": None, "pct_clamped": None, "n_estimates": 0}

    return alpha_df[signal_daily.columns], beta_stats


# ============================================================================
# PIPELINE
# ============================================================================

def main():
    print("=" * 60)
    print("run_engine.py — Phase 3 Step 6: Parity Validation")
    print("=" * 60)

    # ── 1. Load data ─────────────────────────────────────────────
    print("\n[1] Loading data ...")
    df = pd.read_csv(DATA_DIR / "unified_dataset.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    df = df.clip(lower=-0.999999)

    ret = np.log1p(df)
    print(f"    Returns shape: {ret.shape}  ({ret.index[0].date()} to {ret.index[-1].date()})")

    # ── 2. Beta neutralization ───────────────────────────────────
    print("\n[2] Beta neutralization (rolling OLS, window=60) ...")
    mkt          = ret["SPX"]
    residual_ret = pd.DataFrame(index=ret.index, columns=ASSETS, dtype=float)

    for asset in ASSETS:
        y    = ret[asset]
        X    = sm.add_constant(mkt)
        rols = RollingOLS(y, X, window=BETA_WINDOW).fit()
        residual_ret[asset] = y - (rols.params["const"] + rols.params["SPX"] * mkt)

    residual_ret = residual_ret.dropna()
    print(f"    Residual returns shape: {residual_ret.shape}")

    # ── 3. Reversal alpha ─────────────────────────────────────────
    print("\n[3] Reversal alpha (window=5) ...")
    rev_final = build_reversal(residual_ret, window=REV_WINDOW)

    # ── 4. Momentum alpha ─────────────────────────────────────────
    print("\n[4] Momentum alpha (skip=5, window=60) ...")
    mom_final = build_momentum(residual_ret, skip=MOM_SKIP, window=MOM_WINDOW)

    # ── 5. Spread alpha (rolling ADF + rolling z-score) ──────────
    print("\n[5] Spread alpha (rolling OLS + rolling ADF, window=120) ...")
    spread_expanded, spread_raw = build_spread(
        residual_ret,
        pairs        = PAIRS,
        window       = SPREAD_WINDOW,
        adf_pval     = ADF_PVAL,
        halflife_max = HALFLIFE_MAX,
        verbose      = True,
    )

    # ── 5b. Regime classification (DCC-GARCH) ────────────────────
    print("\n[5b] Regime classification (DCC-GARCH) ...")
    try:
        regime_df = compute_regime_signals(residual_ret[ASSETS], spread_raw)
        regime_available = True
        print(f"    Regime labels: {regime_df['regime'].value_counts().to_dict()}")
    except Exception as e:
        print(f"    WARNING: DCC failed ({e}). Using 'normal' for all dates.")
        regime_df        = pd.DataFrame({"regime": "normal"}, index=residual_ret.index)
        regime_available = False

    # ── 6. Covariance matrices (weekly, Ledoit-Wolf) ──────────────
    print("\n[6] Rolling Ledoit-Wolf covariance (window=120, weekly) ...")
    returns_all = residual_ret[ASSETS]
    reb_dates   = returns_all.resample("W-FRI").last().index
    cov_dict    = OrderedDict()

    for date in reb_dates:
        win = returns_all.loc[:date].iloc[-COV_WINDOW:]
        if len(win) < COV_WINDOW:
            continue
        lw  = LedoitWolf().fit(win.values)
        cov_dict[date] = pd.DataFrame(lw.covariance_, index=ASSETS, columns=ASSETS)

    print(f"    Covariance matrices: {len(cov_dict)}")

    # ── 6b. Short-term covariance (window=20, daily, FUTURES only, raw returns) ─
    print("\n[6b] Short-term covariance (window=20, daily, FUTURES only) ...")
    cov_dict_st = OrderedDict()

    for date in df[FUTURES].index:
        win = df[FUTURES].loc[:date].iloc[-ST_COV_WINDOW:]
        if len(win) < ST_COV_WINDOW:
            continue
        lw = LedoitWolf().fit(win.values)
        cov_dict_st[date] = pd.DataFrame(lw.covariance_, index=FUTURES, columns=FUTURES)

    print(f"    Short-term covariance matrices: {len(cov_dict_st)}")

    # ── 7b. FM alignment ─────────────────────────────────────────
    print("\n[7b] Aligning signals to expected-return scale (rolling beta) ...")
    reb_dates_list = sorted(cov_dict.keys())

    alpha_rev_aligned,    _ = align_signal_to_expected_return(
        rev_final[ASSETS],       ret[ASSETS], reb_dates_list)
    alpha_mom_aligned,    _ = align_signal_to_expected_return(
        mom_final[ASSETS],       ret[ASSETS], reb_dates_list)
    alpha_spread_aligned, _ = align_signal_to_expected_return(
        spread_expanded[ASSETS], ret[ASSETS], reb_dates_list)

    alpha_rev_n    = normalize_alpha(alpha_rev_aligned.dropna(how="all"))
    alpha_mom_n    = normalize_alpha(alpha_mom_aligned.dropna(how="all"))
    alpha_spread_n = normalize_alpha(alpha_spread_aligned.dropna(how="all"))

    # ── 7b-st. Short-term reversal alpha (raw returns, FUTURES, daily, regime-gated) ─
    print("\n[7b-st] Short-term reversal (raw, FUTURES, window=3, regime-gated) ...")

    # Raw simple returns, FUTURES only, window=3 — no FM alignment
    rev_raw_st   = -df[FUTURES].rolling(3).sum()
    alpha_rev_st = normalize_alpha(rev_raw_st)

    # Regime gating: use official policy from regime_mapping.get_book_actions()
    # FIX (2026-03-31): replaced hardcoded REV_REGIME_SCALE which was killing
    # crowded (×0.0) and crisis (×0.0), causing the optimizer to carry stale
    # positions on those 184 days instead of computing fresh weights.
    # Official policy keeps all regimes active with variable multipliers,
    # ensuring fresh position computation every day.
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
        scale_rows[date] = get_book_actions(lbl)["short_term"]["alpha_multiplier"]

    rev_scale_series = pd.Series(scale_rows)
    rev_label_series = pd.Series(label_rows)
    alpha_rev_gated  = alpha_rev_st.mul(rev_scale_series, axis=0)

    scale_dist = rev_scale_series.value_counts().sort_index()
    label_dist = rev_label_series.value_counts()
    print(f"    Regime label dist : {label_dist.to_dict()}")
    print(f"    Scale dist        : {scale_dist.to_dict()}")
    print(f"    alpha_rev_gated shape: {alpha_rev_gated.dropna(how='all').shape}")

    # ── 7c. Rolling Sharpe combination — Mom + Spread only ────────
    print("\n[7c] Rolling Sharpe combination (Mom + Spread only) ...")

    res_mom_sa    = Book(
        "mom_sa",    alpha_mom_n,    cov_dict, list(cov_dict.keys()),
        GAMMA, KAPPA, LAMBD, MAX_WEIGHT, TARGET_VOL, EWMA_HALFLIFE, SCALE_MIN, SCALE_MAX,
    ).run(ret[ASSETS])
    res_spread_sa = Book(
        "spread_sa", alpha_spread_n, cov_dict, list(cov_dict.keys()),
        GAMMA, KAPPA, LAMBD, MAX_WEIGHT, TARGET_VOL, EWMA_HALFLIFE, SCALE_MIN, SCALE_MAX,
    ).run(ret[ASSETS])

    pnl_blocks = pd.DataFrame({
        "mom":    res_mom_sa["pnl"],
        "spread": res_spread_sa["pnl"],
    }).dropna(how="all")

    common_idx_a = alpha_mom_n.index.intersection(alpha_spread_n.index)
    EQ2 = pd.Series({"mom": 0.5, "spread": 0.5})

    w_rows = {}
    for date in sorted(common_idx_a):
        past = pnl_blocks[pnl_blocks.index < date].iloc[-SHARPE_ROLL:]
        if len(past) < max(10, SHARPE_ROLL // 4):
            w_rows[date] = EQ2.copy()
            continue
        means    = past.mean()
        stds     = past.std().clip(lower=1e-12)
        sharpes  = (means / stds).clip(lower=0.0)
        total    = sharpes.sum()
        sharpe_w = sharpes / total if total > 1e-12 else EQ2.copy()
        w_rows[date] = (1 - SHRINK_NU) * sharpe_w + SHRINK_NU * EQ2

    w_df_dyn    = pd.DataFrame(w_rows).T
    w_df_smooth = w_df_dyn.ewm(halflife=SMOOTH_HL).mean()
    w_df_smooth = w_df_smooth.div(
        w_df_smooth.sum(axis=1).clip(lower=1e-12), axis=0
    )

    alpha_blocks_n = {"mom": alpha_mom_n, "spread": alpha_spread_n}
    combined_rows  = {}
    for date in sorted(common_idx_a):
        if date not in w_df_smooth.index:
            continue
        w_t = w_df_smooth.loc[date]
        row = np.zeros(len(ASSETS))
        for key, blk in alpha_blocks_n.items():
            if date in blk.index:
                row += float(w_t[key]) * blk.loc[date, ASSETS].fillna(0.0).values
        combined_rows[date] = row

    alpha_combined_aligned = pd.DataFrame(combined_rows, index=ASSETS).T
    alpha_combined_aligned.index = pd.DatetimeIndex(alpha_combined_aligned.index)
    print(f"    Combined alpha shape: {alpha_combined_aligned.dropna(how='all').shape}")

    print("\n[7] Engine-only execution -- baseline decoupled")

    # ── 8. ENGINE: medium_term Book + short_term Book ──────────────────
    print("\n[8] Engine: medium_term Book + short_term Book ...")

    medium_term_book = Book(
        name          = "medium_term",
        alpha_df      = alpha_combined_aligned,
        cov_dict      = cov_dict,
        reb_dates     = list(cov_dict.keys()),
        gamma         = GAMMA,
        kappa         = KAPPA,
        lambd         = LAMBD,
        max_weight    = MAX_WEIGHT,
        target_vol    = TARGET_VOL,
        ewma_halflife = EWMA_HALFLIFE,
        scale_min     = SCALE_MIN,
        scale_max     = SCALE_MAX,
    )


    # ── 10. SHORT-TERM BOOK ───────────────────────────────────────────────────
    print("\n[10] Short-term Book (reversal, daily, FUTURES, regime-gated) ...")

    short_term_book = Book(
        name          = "short_term",
        alpha_df      = alpha_rev_gated,
        cov_dict      = cov_dict_st,
        reb_dates     = list(cov_dict_st.keys()),
        gamma         = ST_GAMMA,
        kappa         = ST_KAPPA,
        lambd         = LAMBD,
        max_weight    = ST_MAX_WEIGHT,
        target_vol    = ST_TARGET_VOL,
        ewma_halflife = ST_EWMA_HL,
        scale_min     = SCALE_MIN,
        scale_max     = SCALE_MAX,
    )

    # FIX (2026-03-31): short-term book uses SIMPLE returns (df[ASSETS]), not log
    # returns (ret[ASSETS]).  For extreme daily commodity moves (NGc1, CLc1),
    # log1p amplifies losses 2-14x vs simple, dominating the PnL mean and
    # flipping the sign from +0.05 to -0.39 (confirmed in debug_d4.py scenarios C/D).
    # Medium-term book is unaffected: it aggregates 5 daily log returns per period,
    # so extreme single-day moves are diluted across the weekly sum.
    res_mt_engine = Allocator([medium_term_book]).run(ret[ASSETS])
    res_st        = short_term_book.run(df[ASSETS])

    pnl_mt_combined = res_mt_engine["book_results"]["medium_term"]["pnl"]
    pnl_st          = res_st["pnl"]
    pnl_engine      = pnl_mt_combined.add(pnl_st, fill_value=0.0)

    # Short-term validity check
    assert len(pnl_st) > 0,         "Short-term book produced empty PnL"
    assert not pnl_st.isna().all(), "Short-term book PnL is all NaN"

    def ann_sharpe(pnl, freq):
        return np.sqrt(freq) * pnl.mean() / pnl.std() if pnl.std() > 1e-12 else float("nan")

    # Medium-term: weekly PnL series → annualise with sqrt(52)
    sharpe_mt = ann_sharpe(pnl_mt_combined, 52)
    # Short-term: daily PnL series → annualise with sqrt(252)
    sharpe_st = ann_sharpe(pnl_st, 252)
    # Combined: determine frequency from PnL index density
    n_days = (pnl_engine.index[-1] - pnl_engine.index[0]).days
    obs_per_day = len(pnl_engine) / max(n_days, 1)
    freq_combined = 252 if obs_per_day > 0.5 else 52
    sharpe_combined = ann_sharpe(pnl_engine, freq_combined)

    print(f"    Medium-term PnL length : {len(pnl_mt_combined)},  Sharpe (sqrt52) : {round(sharpe_mt, 4)}")
    print(f"    Short-term  PnL length : {len(pnl_st)},  Sharpe (sqrt252): {round(sharpe_st, 4)}")
    print(f"    Combined    PnL length : {len(pnl_engine)},  Sharpe          : {round(sharpe_combined, 4)}")
    print("\n  SHORT-TERM VALIDITY CHECK PASSED — non-empty, non-NaN PnL")


if __name__ == "__main__":
    main()
