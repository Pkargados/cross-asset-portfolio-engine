#!/usr/bin/env python
# coding: utf-8
"""
run_baseline.py — Phase 2 pipeline (Phase 1 baseline + Phase 2 improvements).

Runs the full Phase 1 pipeline on unified_dataset.csv and generates
reports/report_phase1.json via report.py.

Pipeline:
  1. Load data (data/raw/unified_dataset.csv)
  2. Log returns
  3. Beta neutralization (rolling OLS, window=60)
  4. Reversal alpha  (rolling-5 sum, CS normalize, MAD winsorize)
  5. Momentum alpha  (shift-5 + rolling-60 sum, CS normalize, MAD winsorize)
  6. Spread alpha    (rolling OLS + rolling ADF + rolling z-score, window=120)
  7. Risk-parity combination (3 alphas)
  8. Ledoit-Wolf covariance (rolling, window=60, weekly)
  9. Chernov optimizer  x_t = (γΣ + κI)^-1 (α_t + κx_{t-1})
 10. PnL computation (net of transaction costs)
 11. Standalone backtests per alpha
 12. generate_report()

Usage:
    cd "Statistical Arbitrage Project"
    python run_baseline.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import OrderedDict
from statsmodels.regression.rolling import RollingOLS
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
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

from engine.backtest.report import generate_report
from engine.portfolio.optimizer import chernov_weights

# ── parameters ──────────────────────────────────────────────────────────────
BETA_WINDOW   = 60
REV_WINDOW    = 5
MOM_WINDOW    = 60
MOM_SKIP      = 5
SPREAD_WINDOW = 120
COV_WINDOW    = 120
ADF_PVAL      = 0.10
HALFLIFE_MAX  = 30   # days — discard spread windows with half-life ≥ 30d (too slow)
SHARPE_ROLL   = 52   # weeks — rolling window for per-block Sharpe estimation
SHRINK_NU     = 0.3  # shrinkage toward equal weights: w = (1-ν)·sharpe_w + ν·(1/3)
SMOOTH_HL     = 4    # EWM halflife (weeks) for smoothing dynamic combination weights
GAMMA         = 20
KAPPA         = 10
LAMBD         = 0.0002
MAX_WEIGHT    = 0.07
TARGET_VOL    = 0.15
EWMA_HALFLIFE = 2    # weeks — EWMA halflife for realized-vol estimation (pnl-based)
SCALE_MIN     = 0.2  # floor on vol-targeting scale
SCALE_MAX     = 1.5  # ceiling on vol-targeting scale
ALIGN_WINDOW  = 52   # weeks for rolling beta: signal → expected return

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
# HELPERS
# ============================================================================

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
    """CS z-score → MAD winsorize (CLAUDE.md §3.6, 2 of 3 steps; re-norm in Phase 2)."""
    return robust_winsorize(cs_normalize(raw))


def optimize_weights(alpha_t, Sigma_t, x_old, n, gamma, kappa, lambd, max_weight):
    """Chernov closed-form L2 optimizer."""
    alpha_t = np.asarray(alpha_t).reshape(-1, 1)
    x_old   = np.asarray(x_old).reshape(-1, 1)
    Sigma   = np.asarray(Sigma_t)

    A = gamma * Sigma + kappa * np.eye(n)
    b = alpha_t + kappa * x_old - lambd * np.sign(x_old)
    x = np.linalg.solve(A, b).flatten()
    x = x - x.mean()                           # dollar neutrality
    x = np.clip(x, -max_weight, max_weight)    # position limits
    return x


def run_backtest(alpha_df, cov_dict, returns_df,
                 gamma=GAMMA, kappa=KAPPA, lambd=LAMBD,
                 max_weight=MAX_WEIGHT, target_vol=TARGET_VOL,
                 ewma_halflife=EWMA_HALFLIFE,
                 scale_min=SCALE_MIN, scale_max=SCALE_MAX):
    """
    Weekly-rebalanced Chernov backtest with EWMA power-scaling vol targeting.

    Risk control layers (in order):
      1. Chernov optimizer  — Ledoit-Wolf Σ for relative position sizing only.
      2. EWMA vol scaling   — rv_t = sqrt(ewma_var_t * 52), updated each period
                              with the just-realized PnL (no look-ahead).
                              scale_raw = (TARGET_VOL / rv_t)^2.0  (power scaling:
                              more aggressive deleveraging above target, faster
                              re-leveraging below).
                              Cap-aware: scale bounded by MAX_WEIGHT / max(|x_raw|)
                              before power clip so no position exceeds MAX_WEIGHT.
                              Final scale = clip(scale_raw, SCALE_MIN, SCALE_MAX).
      3. MAX_WEIGHT clip    — safety net; should rarely bind with cap-aware scaling.

    EWMA initialized at rv ≈ TARGET_VOL (neutral prior → scale = 1 at start).
    Ledoit-Wolf Σ used only for relative sizing; NOT for portfolio vol estimation.

    Returns dict with weights, pnl, sharpe, max_dd, turnover, avg_scale, n_cap_bind.
    """
    reb_dates = sorted(cov_dict.keys())
    alpha_df  = alpha_df.dropna()

    common = (
        pd.DatetimeIndex(reb_dates)
        .intersection(alpha_df.index)
        .intersection(returns_df.index)
    )
    if len(common) < 20:
        return {"pnl": pd.Series(dtype=float), "sharpe": np.nan}

    assets = alpha_df.columns.tolist()
    n      = len(assets)

    # Pre-compute weekly return vectors for every rebalancing period (d, next_d].
    # Used in-loop for EWMA vol update and in post-loop for final PnL.
    common_sorted = sorted(common)
    weekly_ret_map = {}
    for i in range(len(common_sorted) - 1):
        d_curr = common_sorted[i]
        d_next = common_sorted[i + 1]
        mask = (returns_df.index > d_curr) & (returns_df.index <= d_next)
        wret = returns_df.loc[mask, assets]
        if len(wret) > 0:
            weekly_ret_map[d_curr] = wret.values.sum(axis=0)   # shape (n,)

    # EWMA decay factor: alpha = 1 - exp(-ln2 / halflife)
    ewma_alpha = 1.0 - np.exp(-np.log(2.0) / ewma_halflife)
    # Neutral prior: initialize ewma_var so rv ≈ TARGET_VOL → scale = 1 at t=0
    ewma_var   = (target_vol / np.sqrt(52)) ** 2

    weights_dict  = OrderedDict()
    x_prev        = np.zeros(n)
    prev_x        = None   # weights held last period (to compute realized PnL)
    prev_date     = None   # date those weights were set
    scale_history = []     # per-period scale applied (diagnostic)
    cap_bind_hist = []     # per-period cap-constraint indicator (diagnostic)

    for date in common_sorted:
        # ── update EWMA vol with last week's realized PnL ─────────────────────
        # At date d_i, the return for (d_{i-1}, d_i] has just been observed.
        # Update ewma_var BEFORE sizing new positions — no look-ahead.
        if prev_x is not None and prev_date in weekly_ret_map:
            pnl_t  = float(np.dot(prev_x, weekly_ret_map[prev_date]))
            ewma_var = (1.0 - ewma_alpha) * ewma_var + ewma_alpha * pnl_t ** 2

        alpha_t = alpha_df.loc[date, assets]
        Sigma_t = cov_dict[date].loc[assets, assets]

        # ── 1. Optimize (Chernov + Ledoit-Wolf Σ, unchanged) ─────────────────
        x_t = chernov_weights(alpha_t, Sigma_t, x_prev, n,
                              gamma, kappa, lambd, max_weight)

        # ── 2. EWMA power scaling (cap-aware) ─────────────────────────────────
        # rv    = sqrt(ewma_var * 52)  — annualized EWMA realized vol
        # scale_raw = (TARGET_VOL / rv)^2.0  — power scaling (more convex than
        #             linear: deleverages harder above target, re-levers faster below)
        # scale_cap = MAX_WEIGHT / max(|x_raw|) — cap-preserving uniform bound
        # scale     = clip(min(scale_raw, scale_cap), SCALE_MIN, SCALE_MAX)
        # Dollar neutrality preserved: scalar × dollar-neutral = dollar-neutral.
        rv = float(np.sqrt(max(ewma_var, 0.0) * 52))
        scale_applied = 1.0
        cap_bound     = False
        if rv > 1e-8:
            scale_raw = (target_vol / rv) ** 2.0
            max_abs_x = float(np.max(np.abs(x_t)))
            if max_abs_x > 1e-10:
                scale_cap = max_weight / max_abs_x
                if scale_raw > scale_cap:
                    scale_raw = scale_cap
                    cap_bound = True
            scale_applied = float(np.clip(scale_raw, scale_min, scale_max))
            x_t = x_t * scale_applied
        scale_history.append(scale_applied)
        cap_bind_hist.append(cap_bound)

        # ── 3. Hard constraints (safety — should rarely bind with cap-aware scaling)
        x_t = x_t - x_t.mean()                        # dollar neutrality
        x_t = np.clip(x_t, -max_weight, max_weight)   # safety clip

        weights_dict[date] = x_t
        x_prev    = x_t.copy()
        prev_x    = x_t.copy()
        prev_date = date

    w_df = pd.DataFrame(weights_dict, index=assets).T

    # Final PnL: reuse pre-computed weekly_ret_map (no duplication)
    hold_dates = w_df.index.intersection(pd.DatetimeIndex(weekly_ret_map.keys()))
    weekly_ret_rows = {d: pd.Series(v, index=assets) for d, v in weekly_ret_map.items()
                       if d in hold_dates}
    next_ret   = pd.DataFrame(weekly_ret_rows).T
    hold_dates = w_df.index.intersection(next_ret.index)

    w_held       = w_df.loc[hold_dates]
    nr_held      = next_ret.loc[hold_dates]
    gross_pnl    = (w_held.values * nr_held.values).sum(axis=1)
    turnover_s   = w_held.diff().abs().sum(axis=1).fillna(0.0)
    tc_s         = lambd * turnover_s
    pnl          = pd.Series(gross_pnl - tc_s.values, index=hold_dates)

    cumret      = (1 + pnl).cumprod()
    sharpe      = np.sqrt(52) * pnl.mean() / pnl.std() if pnl.std() > 1e-12 else np.nan
    running_max = cumret.cummax()
    max_dd      = float(((cumret - running_max) / running_max).min())

    avg_scale  = float(np.mean(scale_history)) if scale_history else 1.0
    n_cap_bind = int(sum(cap_bind_hist))

    return {
        "weights":    w_held,
        "pnl":        pnl,
        "sharpe":     round(float(sharpe), 4),
        "max_dd":     round(max_dd, 4),
        "turnover":   round(float(turnover_s.mean()), 6),
        "avg_scale":  round(avg_scale, 4),
        "n_cap_bind": n_cap_bind,
    }


def align_signal_to_expected_return(
    signal_daily: pd.DataFrame,
    returns_daily: pd.DataFrame,
    reb_dates,
    window: int = ALIGN_WINDOW,
    beta_floor: float = 0.0,
) -> tuple:
    """
    Scale a CS-normalized signal to expected-return units via rolling Fama-MacBeth.

    At each rebalancing Friday t:
      For each past week τ in [t-window, t-1], run a CROSS-SECTIONAL OLS:
          r_{i, τ+1} = a + β_τ × signal_{i, τ}    (across i = 1..N assets)
      Take β̄_t = mean(β_{t-W}, ..., β_{t-1}).
      α_{i,t} = max(beta_floor, β̄_t) × signal_{i,t}

    Parameters
    ----------
    beta_floor : float
        Floor applied to raw β̄ before scaling.
        Default 0.0 (clamp at 0): prevents inverting the signal direction when
        the FM estimate turns negative (estimation noise in low-N cross-sections).
        Set to -np.inf to remove the floor entirely (for diagnostic purposes).

    Returns
    -------
    (alpha_df, beta_stats) where beta_stats = {
        "raw_mean": mean of all raw β̄_t estimates,
        "raw_std":  std,
        "pct_negative": fraction of estimates that were < 0,
        "pct_clamped":  fraction that were changed by the floor
    }

    No look-ahead bias: β̄ uses only fwd_ret strictly up to t-1.
    """
    reb_sorted = sorted(reb_dates)
    assets = [a for a in signal_daily.columns if a in returns_daily.columns]

    # Weekly forward returns: fwd_ret[d_curr] = log ret from (d_curr, d_next]
    fwd_rows = {}
    for i in range(len(reb_sorted) - 1):
        d_curr, d_next = reb_sorted[i], reb_sorted[i + 1]
        mask = (returns_daily.index > d_curr) & (returns_daily.index <= d_next)
        wret = returns_daily.loc[mask, assets]
        if len(wret) > 0:
            fwd_rows[d_curr] = wret.sum(axis=0)

    fwd_df = pd.DataFrame(fwd_rows).T  # (n_weeks, N)

    # Signal sampled at each Friday (forward-fill for holiday gaps)
    sig_weekly = signal_daily[assets].reindex(fwd_df.index, method="ffill")

    s_mat = sig_weekly.values   # (T, N)
    r_mat = fwd_df.values       # (T, N)  fwd_ret[t] = return AFTER Friday t
    T, N  = s_mat.shape
    alpha_vals = np.full((T, N), np.nan)

    raw_betas = []   # track raw β̄ for diagnostics

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

    # Beta diagnostics
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
    print("Phase 2 — Statistical Arbitrage Pipeline")
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
    mkt         = ret["SPX"]
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
    rev_raw   = -residual_ret.rolling(REV_WINDOW).sum()
    rev_final = normalize_alpha(rev_raw)
    print(f"    Shape: {rev_final.dropna().shape}")

    # ── 4. Momentum alpha ─────────────────────────────────────────
    print("\n[4] Momentum alpha (skip=5, window=60) ...")
    mom_raw   = residual_ret.shift(MOM_SKIP).rolling(MOM_WINDOW).sum()
    mom_final = normalize_alpha(mom_raw)
    print(f"    Shape: {mom_final.dropna().shape}")

    # ── 5. Spread alpha (rolling ADF + rolling z-score) ──────────
    print("\n[5] Spread alpha (rolling OLS + rolling ADF, window=120) ...")
    pair_labels = [f"{e}_{c}" for e, c in PAIRS]
    spread_raw  = pd.DataFrame(index=residual_ret.index,
                               columns=pair_labels, dtype=float)

    for idx, (etf, com) in enumerate(PAIRS):
        print(f"    Pair {idx+1}/{len(PAIRS)}: {etf}-{com} ...", end=" ", flush=True)
        y    = residual_ret[etf]
        X    = residual_ret[com]
        rols = RollingOLS(y, sm.add_constant(X), window=SPREAD_WINDOW).fit()
        spread = y - (rols.params["const"] + rols.params[com] * X)

        spread_arr = spread.values
        signal     = np.zeros(len(spread_arr))

        n_active   = 0
        half_lives = []
        for i in range(SPREAD_WINDOW - 1, len(spread_arr)):
            w_data  = spread_arr[i - SPREAD_WINDOW + 1 : i + 1]
            w_clean = w_data[~np.isnan(w_data)]
            if len(w_clean) < SPREAD_WINDOW // 2:
                continue
            try:
                pval = adfuller(w_clean)[1]
            except Exception:
                pval = 1.0
            if pval >= ADF_PVAL:
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
            # ρ must be in (-1, 0) for a finite, positive half-life
            if rho >= 0.0 or (1.0 + rho) <= 0.0:
                continue
            half_life = -np.log(2.0) / np.log(1.0 + rho)
            if half_life >= HALFLIFE_MAX:
                continue
            # ECM used as pure filter: ADF + (ρ<0, half_life<30d) gates the window.
            # No 1/half_life scaling: estimation error in ρ near 0 dominates the
            # scaling term in a 120-day window; CS normalization downstream handles
            # cross-pair weighting.
            half_lives.append(half_life)
            signal[i] = -(spread_arr[i] - mu) / std
            n_active += 1

        avg_hl = float(np.mean(half_lives)) if half_lives else float("nan")
        spread_raw[f"{etf}_{com}"] = pd.Series(signal, index=spread.index)
        print(f"active={n_active} windows  avg_hl={avg_hl:.1f}d")

    # Normalize spread pair signals
    spread_norm  = cs_normalize(spread_raw.astype(float))
    spread_wins  = robust_winsorize(spread_norm)
    spread_final = spread_wins

    # Map pair signals → asset columns (both ETF and commodity legs).
    # Spread = ETF_resid - (a + b·COM_resid) → signal = -(spread-mu)/std.
    # Positive signal → ETF below predicted value → buy ETF (+), sell commodity (-).
    # Assigning -signal to the commodity leg restores full pair structure and gives
    # futures non-zero spread exposure (they were zero before this change).
    spread_expanded = pd.DataFrame(0.0,
                                   index=residual_ret.index,
                                   columns=residual_ret.columns)
    pair_count_exp = {a: 0 for a in ASSETS}
    for etf, com in PAIRS:
        sig = spread_final[f"{etf}_{com}"].fillna(0.0)
        spread_expanded[etf] += sig
        spread_expanded[com] -= sig
        pair_count_exp[etf] += 1
        pair_count_exp[com] += 1
    for asset in ASSETS:
        if pair_count_exp[asset] > 1:
            spread_expanded[asset] /= pair_count_exp[asset]
    print(f"    Expanded spread shape: {spread_expanded.shape}")
    n_nonzero_assets = int((spread_expanded.abs().sum() > 0).sum())
    print(f"    Assets with non-zero spread signal: {n_nonzero_assets}/{len(ASSETS)}")

    # Raw expanded spread (same structure, for FM alignment)
    spread_expanded_raw = pd.DataFrame(0.0,
                                       index=residual_ret.index,
                                       columns=residual_ret.columns)
    pair_count_raw = {a: 0 for a in ASSETS}
    for etf, com in PAIRS:
        sig_r = spread_raw[f"{etf}_{com}"].fillna(0.0)
        spread_expanded_raw[etf] += sig_r
        spread_expanded_raw[com] -= sig_r
        pair_count_raw[etf] += 1
        pair_count_raw[com] += 1
    for asset in ASSETS:
        if pair_count_raw[asset] > 1:
            spread_expanded_raw[asset] /= pair_count_raw[asset]

    # ── 6. Risk-parity combination ────────────────────────────────
    print("\n[6] Risk-parity combination (3 blocks) ...")
    vol_rev    = rev_final.std().mean()
    vol_mom    = mom_final.std().mean()
    vol_spread = spread_expanded.std().mean()

    raw_w = np.array([1 / vol_rev, 1 / vol_mom, 1 / vol_spread])
    w     = raw_w / raw_w.sum()
    weights_rp = {"reversal": w[0], "momentum": w[1], "spreads": w[2]}
    print(f"    Weights → rev={w[0]:.3f}  mom={w[1]:.3f}  spread={w[2]:.3f}")

    common_idx = (
        rev_final.index
        .intersection(mom_final.index)
        .intersection(spread_expanded.index)
    )
    common_cols = rev_final.columns

    alpha_rev    = rev_final.loc[common_idx, common_cols]
    alpha_mom    = mom_final.loc[common_idx, common_cols]
    alpha_spread = spread_expanded.loc[common_idx, common_cols]

    alpha_combined = (
          w[0] * alpha_rev
        + w[1] * alpha_mom
        + w[2] * alpha_spread
    )
    print(f"    Combined alpha shape: {alpha_combined.dropna().shape}")

    # ── 7. Covariance matrices (weekly, Ledoit-Wolf) ──────────────
    # Use residual_ret (beta-neutralized) so the risk model matches the return
    # process that generates PnL. Using raw ret[ASSETS] caused vol targeting to
    # mis-scale positions (Phase 1: 55.8% realized vol vs. 15% target).
    print("\n[7] Rolling Ledoit-Wolf covariance (window=60, weekly) ...")
    returns_all   = residual_ret[ASSETS]
    reb_dates     = returns_all.resample("W-FRI").last().index
    cov_dict      = OrderedDict()

    for date in reb_dates:
        win = returns_all.loc[:date].iloc[-COV_WINDOW:]
        if len(win) < COV_WINDOW:
            continue
        lw  = LedoitWolf().fit(win.values)
        cov_dict[date] = pd.DataFrame(lw.covariance_, index=ASSETS, columns=ASSETS)

    print(f"    Covariance matrices: {len(cov_dict)}")

    # ── 7b. Align signals to expected-return scale ────────────────
    print("\n[7b] Aligning signals to expected-return scale (rolling β) ...")
    reb_dates_list = sorted(cov_dict.keys())

    # Pass CS-normalized signals: ranking is cross-sectional, β̄ scales to E[r] units.
    # beta_floor=0.0 (default): prevents FM from inverting the signal when noise drives
    # β̄ negative.  For signals pre-signed positive (reversal negated, momentum direct,
    # spread mean-reversion correct), a negative β̄ reflects estimation noise, not a
    # genuine reversal of the economic premise.
    alpha_rev_aligned,    bstats_rev    = align_signal_to_expected_return(
        rev_final[ASSETS],        ret[ASSETS], reb_dates_list)
    alpha_mom_aligned,    bstats_mom    = align_signal_to_expected_return(
        mom_final[ASSETS],        ret[ASSETS], reb_dates_list)
    alpha_spread_aligned, bstats_spread = align_signal_to_expected_return(
        spread_expanded[ASSETS],  ret[ASSETS], reb_dates_list)

    print(f"    Rev aligned:    {alpha_rev_aligned.dropna(how='all').shape}")
    print(f"    Mom aligned:    {alpha_mom_aligned.dropna(how='all').shape}")
    print(f"    Spread aligned: {alpha_spread_aligned.dropna(how='all').shape}")

    # ── FM β diagnostics ──────────────────────────────────────────
    print("\n    FM β̄ diagnostics (beta_floor=0.0 — clamp in effect):")
    for name, bs in [("Rev", bstats_rev), ("Mom", bstats_mom), ("Spread", bstats_spread)]:
        print(f"      {name:8s}: β̄_mean={bs['raw_mean']:+.5f}  β̄_std={bs['raw_std']:.5f}"
              f"  pct_negative={bs['pct_negative']:.1%}  "
              f"pct_clamped_to_0={bs['pct_clamped']:.1%}  n={bs['n_estimates']}")

    # ── β ≥ 0 investigation: reversal with unclamped FM beta ─────
    # Reversal signal is pre-signed: rev_raw = -rolling_sum → beaten-down assets
    # get high signal → FM β should be positive if reversal works.
    # With IC ≈ 0, β̄ ≈ 0 and ~50% of estimates are negative → clamp zeros them out.
    # Question: does allowing negative β (which turns those weeks into trend-following)
    # improve IC or Sharpe, or does it add noise and hurt performance?
    print("\n    [7b-inv] β ≥ 0 investigation: reversal with beta_floor=-inf ...")
    from engine.backtest.report import compute_ic as _compute_ic
    alpha_rev_free, bstats_rev_free = align_signal_to_expected_return(
        rev_final[ASSETS], ret[ASSETS], reb_dates_list, beta_floor=-np.inf)
    alpha_rev_free_n = normalize_alpha(alpha_rev_free.dropna(how="all"))

    # Use pre-normalized aligned alpha for clamped IC (comparable to free version)
    alpha_rev_clamped_n = normalize_alpha(alpha_rev_aligned.dropna(how="all"))
    ic_clamped  = _compute_ic(alpha_rev_clamped_n, ret[ASSETS])
    ic_free     = _compute_ic(alpha_rev_free_n,    ret[ASSETS])

    res_rev_clamped_inv = run_backtest(alpha_rev_clamped_n, cov_dict, ret[ASSETS])
    res_rev_free_inv    = run_backtest(alpha_rev_free_n,    cov_dict, ret[ASSETS])

    print(f"      Clamped  (β≥0): IC={ic_clamped['mean_ic']:+.5f}  t={ic_clamped['ic_tstat']:+.3f}"
          f"  standalone Sharpe={res_rev_clamped_inv['sharpe']:+.4f}")
    print(f"      Unclamped(free): IC={ic_free['mean_ic']:+.5f}  t={ic_free['ic_tstat']:+.3f}"
          f"  standalone Sharpe={res_rev_free_inv['sharpe']:+.4f}")
    print(f"      Raw β̄ distribution (both runs same): "
          f"mean={bstats_rev_free['raw_mean']:+.5f}  "
          f"std={bstats_rev_free['raw_std']:.5f}  "
          f"pct_negative={bstats_rev_free['pct_negative']:.1%}")

    # Decision: keep β≥0 clamp for reversal unless unclamped shows clear improvement.
    # Rationale: negative β̄ → FM estimated beaten-down assets underperform → inverting
    # the signal converts mean-reversion to trend-following for those weeks, adding noise.
    # A weaker but consistently directional signal is preferable to an adaptive one that
    # violates the mean-reversion economic premise.
    # The rolling Sharpe combination (step 7c) will naturally down-weight reversal if
    # its standalone performance remains poor — this is the correct regime-adaptive fix.
    print("      → Decision: retaining β≥0 clamp for all signals (see memory.md §FM β diagnostic)")

    # ── Re-normalize aligned alphas (CS z-score + MAD winsorize) ──
    # After FM β̄ scaling, each block's magnitude ∝ β̄_block × signal_std.
    # β̄ values differ across signals (proportional to each signal's IC), so
    # without re-normalization the risk-parity vol denominator just inverts
    # each β̄ — making combination weights arbitrary and destabilizing the
    # optimizer when any one β̄ is near zero.
    # Re-normalizing restores a common, comparable scale before weighting.

    def _std_summary(df):
        vals = df.stack().dropna()
        return f"{vals.std():.2e}" if len(vals) else "N/A"

    print(f"    FM-scaled std (pre-norm): "
          f"rev={_std_summary(alpha_rev_aligned)}  "
          f"mom={_std_summary(alpha_mom_aligned)}  "
          f"spread={_std_summary(alpha_spread_aligned)}")

    alpha_rev_n    = normalize_alpha(alpha_rev_aligned.dropna(how="all"))
    alpha_mom_n    = normalize_alpha(alpha_mom_aligned.dropna(how="all"))
    alpha_spread_n = normalize_alpha(alpha_spread_aligned.dropna(how="all"))

    print(f"    Re-normalized std (post): "
          f"rev={_std_summary(alpha_rev_n)}  "
          f"mom={_std_summary(alpha_mom_n)}  "
          f"spread={_std_summary(alpha_spread_n)}")

    # ── 7c. Rolling Sharpe combination (Phase 2 Step 2.1) ────────
    # MEDIUM-TERM BOOK ONLY: Momentum + Spread.
    # Reversal (5-day horizon) is excluded from this combination — its horizon
    # is incompatible with the weekly optimizer (κ=10 inertia, weekly rebalance).
    # Rev standalone backtest is still run below for attribution and diagnostics.
    # Reversal will be reintegrated in a dedicated short-term book in a future phase.
    print("\n[7c] Rolling Sharpe combination — Mom + Spread only "
          f"(window={SHARPE_ROLL}w, ν={SHRINK_NU}, hl={SMOOTH_HL}w) ...")

    # Standalone backtests — rev kept for attribution; mom+spread drive combination
    res_rev    = run_backtest(alpha_rev_n,    cov_dict, ret[ASSETS])
    res_mom    = run_backtest(alpha_mom_n,    cov_dict, ret[ASSETS])
    res_spread = run_backtest(alpha_spread_n, cov_dict, ret[ASSETS])

    # Only mom and spread enter the combination PnL pool
    pnl_blocks = pd.DataFrame({
        "mom":    res_mom["pnl"],
        "spread": res_spread["pnl"],
    }).dropna(how="all")

    common_idx_a = (
        alpha_mom_n.index
        .intersection(alpha_spread_n.index)
    )
    EQ2 = pd.Series({"mom": 0.5, "spread": 0.5})

    w_rows = {}
    for date in sorted(common_idx_a):
        # Strict past-only: PnL from before this rebalancing date
        past = pnl_blocks[pnl_blocks.index < date].iloc[-SHARPE_ROLL:]
        if len(past) < max(10, SHARPE_ROLL // 4):
            w_rows[date] = EQ2.copy()
            continue
        means    = past.mean()
        stds     = past.std().clip(lower=1e-12)
        sharpes  = (means / stds).clip(lower=0.0)   # floor negative Sharpe at 0
        total    = sharpes.sum()
        sharpe_w = sharpes / total if total > 1e-12 else EQ2.copy()
        # Shrink toward equal weights
        w_rows[date] = (1 - SHRINK_NU) * sharpe_w + SHRINK_NU * EQ2

    w_df_dyn = pd.DataFrame(w_rows).T   # (n_dates, 2) — raw Sharpe weights
    # EWM smoothing, then re-normalize so weights sum to 1
    w_df_smooth = w_df_dyn.ewm(halflife=SMOOTH_HL).mean()
    w_df_smooth = w_df_smooth.div(
        w_df_smooth.sum(axis=1).clip(lower=1e-12), axis=0
    )

    print(f"    Dynamic weight stats (post-smooth):")
    for col in ["mom", "spread"]:
        s = w_df_smooth[col]
        print(f"      {col:8s}: mean={s.mean():.3f}  std={s.std():.3f}  "
              f"min={s.min():.3f}  max={s.max():.3f}")

    # Build time-varying combined alpha (mom + spread only)
    alpha_blocks_n = {
        "mom":    alpha_mom_n,
        "spread": alpha_spread_n,
    }
    combined_rows = {}
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
    print(f"    Combined alpha shape (Mom+Spread): "
          f"{alpha_combined_aligned.dropna(how='all').shape}")

    # Save re-normalized alpha pkl files
    ALPHA_DIR = ROOT / "data" / "alphas"
    ALPHA_DIR.mkdir(parents=True, exist_ok=True)
    alpha_rev_n.to_pickle(ALPHA_DIR / "alpha_rev.pkl")
    alpha_mom_n.to_pickle(ALPHA_DIR / "alpha_mom.pkl")
    alpha_spread_n.to_pickle(ALPHA_DIR / "alpha_spread.pkl")
    alpha_combined_aligned.to_pickle(ALPHA_DIR / "alpha_combined.pkl")
    print(f"    Alpha files saved → {ALPHA_DIR}")

    # ── 7d. Regime detection (DCC-GARCH) ─────────────────────────
    # DCC is fit once on full sample; R_t is the recursively-filtered correlation
    # path (parameters have minor full-sample look-ahead — standard practice).
    # alpha_combined_final = alpha_combined_aligned × scale_t per rebalancing date.
    print("\n[7d] DCC-GARCH regime detection ...")
    regime_available  = False
    regime_df         = None
    scale_series      = None
    dcc_converged_flag = None
    try:
        from engine.regime.regime_detection import compute_regime_signals, compute_alpha_scale
        regime_df    = compute_regime_signals(ret[ASSETS], spread_raw)
        print(regime_df["regime"].value_counts())
        scale_series = compute_alpha_scale(regime_df, alpha_combined_aligned.index)
        alpha_combined_final = alpha_combined_aligned.multiply(
            scale_series.reindex(alpha_combined_aligned.index).fillna(1.0), axis=0
        )
        print(f"    Regime-adjusted alpha shape: "
              f"{alpha_combined_final.dropna(how='all').shape}")
        print(f"    Regime stats — rho: mean={regime_df['rho_t'].mean():.3f}  "
              f"std={regime_df['rho_t'].std():.3f}  "
              f"n_spikes={int(regime_df['spike_t'].sum())}")
        print(f"    Scale stats  — mean={scale_series.mean():.3f}  "
              f"min={scale_series.min():.3f}  "
              f"n_reduced={int((scale_series < 1.0).sum())}/{len(scale_series)}")
        regime_available   = True
        dcc_converged_flag = True
    except Exception as _dcc_err:
        print(f"    [WARNING] DCC failed: {_dcc_err} — proceeding without regime adjustment")
        alpha_combined_final = alpha_combined_aligned

    # ── 8. Backtests ──────────────────────────────────────────────
    # Standalone results (res_rev, res_mom, res_spread) already computed in [7c].
    # Only the combined portfolio needs a fresh run with the dynamic-weight alpha.
    print("\n[8] Running combined backtest ...")

    res_combined = run_backtest(alpha_combined_final, cov_dict, ret[ASSETS])

    print(f"    Combined  Sharpe={res_combined['sharpe']:+.4f}  MaxDD={res_combined['max_dd']:+.4f}"
          f"  avg_scale={res_combined.get('avg_scale', 'N/A')}  n_cap_bind={res_combined.get('n_cap_bind', 'N/A')}")
    print(f"    Rev       Sharpe={res_rev['sharpe']:+.4f}  MaxDD={res_rev['max_dd']:+.4f}")
    print(f"    Mom       Sharpe={res_mom['sharpe']:+.4f}  MaxDD={res_mom['max_dd']:+.4f}")
    print(f"    Spread    Sharpe={res_spread['sharpe']:+.4f}  MaxDD={res_spread['max_dd']:+.4f}")

    # ── 9. Generate report ────────────────────────────────────────
    print("\n[9] Generating report ...")

    alpha_dict_report = {
        "Rev":    alpha_rev_n,
        "Mom":    alpha_mom_n,
        "Spread": alpha_spread_n,
    }
    standalone = {
        "Rev":    {"pnl": res_rev["pnl"],    "sharpe": res_rev["sharpe"],    "max_dd": res_rev["max_dd"]},
        "Mom":    {"pnl": res_mom["pnl"],    "sharpe": res_mom["sharpe"],    "max_dd": res_mom["max_dd"]},
        "Spread": {"pnl": res_spread["pnl"], "sharpe": res_spread["sharpe"], "max_dd": res_spread["max_dd"]},
    }

    report = generate_report(
        pnl           = res_combined["pnl"],
        returns_df    = ret[ASSETS],
        alpha_dict    = alpha_dict_report,
        standalone    = standalone,
        weights_df    = res_combined.get("weights"),
        phase_name    = "Phase2c",
        regime_df     = regime_df,
        scale_series  = scale_series,
        dcc_converged = dcc_converged_flag,
        description   = (
            "Phase 2c — EWMA power-scaling vol targeting with cap-aware scaling. "
            "rv=sqrt(ewma_var*52), halflife=2w, neutral prior rv0=TARGET_VOL. "
            "scale_raw=(TARGET_VOL/rv)^2.0; scale_cap=MAX_WEIGHT/max(|x_raw|); "
            "scale=clip(min(scale_raw,scale_cap), SCALE_MIN=0.2, SCALE_MAX=1.5). "
            "MAX_WEIGHT=0.07; LW Σ retained in optimizer for relative sizing only. "
            "Medium-term book: Momentum + Spread only (reversal excluded, horizon mismatch). "
            "2.2a ECM filter: rolling 120d; discard if rho>=0 or half-life>=30d. "
            "2.2b Both pair legs signalled. "
            "2.1 Rolling Sharpe combination (Mom+Spread): 52w, nu=0.3, hl=4w EWM. "
            "Covariance: Ledoit-Wolf on residual_ret (beta-neutralized). "
            "2.3 DCC-GARCH regime conditioning: "
            + ("applied" if regime_available else "skipped (DCC unavailable)") + ". "
            "DCC fit once on full sample; R_t recursively filtered (Option A). "
            "alpha_scale in [0.1,1.0]: spike_t*0.5 + rho_z taper."
        ),
        output_dir  = str(ROOT / "reports"),
    )

    def _fmt(val, fmt="+.4f"):
        return format(val, fmt) if val is not None else "N/A"

    print("\n" + "=" * 60)
    print("BASELINE REPORT SUMMARY")
    print("=" * 60)
    perf = report["performance"]
    print(f"  Annualized Return : {_fmt(perf['annualized_return'])}")
    print(f"  Annualized Vol    : {_fmt(perf['annualized_volatility'], '.4f')}")
    print(f"  Sharpe Ratio      : {_fmt(perf['sharpe_ratio'])}")
    print(f"  Max Drawdown      : {_fmt(perf['max_drawdown'])}")
    print(f"  Calmar Ratio      : {_fmt(perf['calmar_ratio'])}")
    print(f"  Periods           : {perf['n_periods']}")
    print()
    print("  Signal Quality (IC):")
    for name, sq in report["signal_quality"].items():
        print(f"    {name:8s} mean_IC={sq.get('mean_ic')}  t-stat={sq.get('ic_tstat')}")
    print()
    print("  Stability:")
    st = report["stability"]
    print(f"    Avg Rolling Sharpe : {st['avg_rolling_sharpe']}")
    print(f"    Avg Turnover       : {st['avg_turnover']}")
    print()
    print("  Attribution (standalone Sharpe):")
    for name, attr in report["attribution"].items():
        print(f"    {name:8s} Sharpe={attr.get('sharpe_ratio')}  MaxDD={attr.get('max_dd')}")
    print()
    print("  Diagnostics:")
    diag = report["diagnostics"]
    ge = diag.get("gross_exposure") or {}
    print(f"    Gross Exposure (mean) : {ge.get('mean')}")
    ss = diag.get("spread_stats") or {}
    print(f"    Spread pct_nonzero    : {ss.get('pct_nonzero')}  n_nonzero={ss.get('n_nonzero')}")
    print()
    print("  Regime (DCC):")
    reg = report.get("regime") or {}
    if reg:
        print(f"    Converged         : {reg.get('dcc_converged')}")
        print(f"    rho mean/std      : {reg.get('rho_mean')} / {reg.get('rho_std')}")
        print(f"    n_spikes          : {reg.get('n_spikes')}  ({reg.get('spike_pct', 0)*100:.1f}%)")
        print(f"    scale mean/min    : {reg.get('scale_mean')} / {reg.get('scale_min')}")
        print(f"    n_periods_reduced : {reg.get('n_periods_reduced')}  ({reg.get('pct_periods_reduced', 0)*100:.1f}%)")
    else:
        print("    DCC unavailable — regime section not populated")
    print("=" * 60)

    # ── 10. Volatility targeting diagnosis ───────────────────────
    print("\n[10] Volatility targeting diagnosis ...")
    w_combined   = res_combined.get("weights")
    pnl_combined = res_combined["pnl"]

    if w_combined is not None and len(w_combined) > 5:
        # Model-implied annualized vol at each rebalancing date (from Ledoit-Wolf Σ)
        model_vol_rows = {}
        for date in w_combined.index:
            if date in cov_dict:
                x_t     = w_combined.loc[date, ASSETS].values
                Sigma_t = cov_dict[date].loc[ASSETS, ASSETS].values
                port_var = x_t @ Sigma_t @ x_t
                model_vol_rows[date] = float(np.sqrt(max(port_var, 0) * 252))
        model_vol = pd.Series(model_vol_rows)

        # Realized vol: 8-week rolling std of weekly PnL, annualized with sqrt(52)
        realized_vol = pnl_combined.rolling(8).std() * np.sqrt(52)

        common_vol = model_vol.index.intersection(realized_vol.dropna().index)
        if len(common_vol) > 5:
            mv    = model_vol.loc[common_vol]
            rv    = realized_vol.loc[common_vol]
            ratio = rv / mv.clip(lower=1e-8)
            print(f"\n  Model-implied vol  (mean): {mv.mean():.4f}  ({mv.mean()*100:.1f}%)")
            print(f"  Realized vol       (mean): {rv.mean():.4f}  ({rv.mean()*100:.1f}%)")
            print(f"  Realized / Model   ratio : {ratio.mean():.2f}x  (std={ratio.std():.2f})")
            print(f"  Ratio >2x pct            : {(ratio > 2).mean():.2%}")

        # Per-asset marginal variance contribution at last date
        last_date = w_combined.index[-1]
        if last_date in cov_dict:
            x_last     = w_combined.loc[last_date, ASSETS].values
            Sigma_last = cov_dict[last_date].loc[ASSETS, ASSETS].values
            pv_last    = max(x_last @ Sigma_last @ x_last, 1e-16)
            mc         = (Sigma_last @ x_last) * x_last / pv_last
            mc_series  = pd.Series(mc, index=ASSETS).sort_values(ascending=False)
            print(f"\n  Per-asset variance contribution (at {last_date.date()}):")
            for asset, contrib in mc_series.items():
                print(f"    {asset:8s}  {contrib:+.4f}")

        # Stability of Σ: average pairwise correlation implied by Ledoit-Wolf
        corr_diag = []
        for date, Sigma in list(cov_dict.items())[-52:]:   # last year
            d_inv = 1.0 / np.sqrt(np.diag(Sigma.values).clip(1e-16))
            R     = Sigma.values * np.outer(d_inv, d_inv)
            n     = len(R)
            off   = R[np.triu_indices(n, k=1)]
            corr_diag.append(off.mean())
        print(f"\n  Avg pairwise correlation (last 52w): {np.mean(corr_diag):.4f}  (std={np.std(corr_diag):.4f})")
        print("=" * 60)


if __name__ == "__main__":
    main()
