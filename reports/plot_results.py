#!/usr/bin/env python
# coding: utf-8
"""
reports/plot_results.py — Phase 4: GitHub Presentation Figures

Generates clean, minimal figures from engine outputs for the README.
Does not modify any engine or strategy files.

Usage (from project root):
    python reports/plot_results.py

Output:
    reports/figures/cumulative_pnl.png
    reports/figures/drawdown.png
    reports/figures/rolling_sharpe.png
    reports/figures/rolling_ic.png
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pathlib import Path
from collections import OrderedDict
from statsmodels.regression.rolling import RollingOLS
import statsmodels.api as sm
from sklearn.covariance import LedoitWolf


# ── project root ──────────────────────────────────────────────────────────────
def _find_project_root() -> Path:
    for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]:
        if (p / "CLAUDE.md").exists():
            return p
    raise RuntimeError("Project root (CLAUDE.md) not found")

ROOT    = _find_project_root()
FIG_DIR = ROOT / "reports" / "figures"
sys.path.insert(0, str(ROOT))

# ── engine imports ─────────────────────────────────────────────────────────────
from engine.portfolio.book          import Book
from engine.portfolio.allocator     import Allocator
from engine.alphas.momentum         import build_momentum
from engine.alphas.spread           import build_spread
from engine.alphas.reversal         import build_reversal
from engine.regime.regime_detection import compute_regime_signals
from engine.regime.regime_mapping   import get_book_actions

# ── strategy constants and helpers (read-only import, main() never called) ────
from strategies.medium_term_rv.run_engine import (
    DATA_DIR,
    FUTURES, ETFS, ASSETS, PAIRS,
    BETA_WINDOW, REV_WINDOW, MOM_WINDOW, MOM_SKIP, SPREAD_WINDOW,
    COV_WINDOW, ADF_PVAL, HALFLIFE_MAX,
    SHARPE_ROLL, SHRINK_NU, SMOOTH_HL,
    GAMMA, KAPPA, LAMBD, MAX_WEIGHT, TARGET_VOL, EWMA_HALFLIFE, SCALE_MIN, SCALE_MAX,
    ST_COV_WINDOW, ST_GAMMA, ST_KAPPA, ST_MAX_WEIGHT, ST_TARGET_VOL, ST_EWMA_HL,
    align_signal_to_expected_return, normalize_alpha,
)

# ── plot style ────────────────────────────────────────────────────────────────
C_COMBINED = "#1f77b4"   # blue  — combined strategy
C_MT_ONLY  = "#aec7e8"   # light blue — medium-term only
C_MOMENTUM = "#2ca02c"   # green — momentum standalone
C_SPREAD   = "#ff7f0e"   # orange — spread standalone
C_DD       = "#d62728"   # red — drawdown fill
FIGSIZE    = (11, 4)

plt.rcParams.update({
    "figure.dpi":        150,
    "font.size":         10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
    "legend.frameon":    False,
    "legend.fontsize":   9,
})


# ============================================================================
# PIPELINE  (reads data + runs Books — no engine files modified)
# ============================================================================

def _rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation (no scipy dependency)."""
    return pd.Series(a).rank().corr(pd.Series(b).rank())


def _compute_rolling_ic(
    alpha_df:   pd.DataFrame,
    returns_df: pd.DataFrame,
    reb_dates:  list,
    window:     int = 26,
) -> pd.Series:
    """
    Rolling mean IC: per-date Spearman rank correlation between alpha
    and subsequent weekly return, smoothed over `window` periods.
    """
    reb_sorted = sorted(reb_dates)
    fwd_rows   = {}
    for i in range(len(reb_sorted) - 1):
        d_curr, d_next = reb_sorted[i], reb_sorted[i + 1]
        mask = (returns_df.index > d_curr) & (returns_df.index <= d_next)
        wret = returns_df.loc[mask]
        if len(wret) > 0:
            fwd_rows[d_curr] = wret.sum(axis=0)

    fwd_df = pd.DataFrame(fwd_rows).T
    common  = alpha_df.index.intersection(fwd_df.index)
    ic_vals = {}
    for date in common:
        a     = alpha_df.loc[date]
        r     = fwd_df.loc[date, a.index]
        valid = ~(a.isna() | r.isna())
        if valid.sum() < 5:
            continue
        ic_vals[date] = _rank_corr(a[valid].values, r[valid].values)

    ic_series = pd.Series(ic_vals).sort_index()
    return ic_series.rolling(window, min_periods=window // 2).mean()


def _run_pipeline() -> dict:
    """
    Run the full engine pipeline and return PnL series for plotting.
    Mirrors strategies/medium_term_rv/run_engine.py exactly;
    no engine or strategy files are modified.
    """
    print("[plot] Loading data ...")
    df = pd.read_csv(DATA_DIR / "unified_dataset.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    df = df.clip(lower=-0.999999)
    ret = np.log1p(df)

    # Beta neutralization
    print("[plot] Beta neutralization ...")
    mkt          = ret["SPX"]
    residual_ret = pd.DataFrame(index=ret.index, columns=ASSETS, dtype=float)
    for asset in ASSETS:
        y    = ret[asset]
        X    = sm.add_constant(mkt)
        rols = RollingOLS(y, X, window=BETA_WINDOW).fit()
        residual_ret[asset] = y - (rols.params["const"] + rols.params["SPX"] * mkt)
    residual_ret = residual_ret.dropna()

    # Alpha construction
    print("[plot] Building alphas ...")
    rev_final                  = build_reversal(residual_ret, window=REV_WINDOW)
    mom_final                  = build_momentum(residual_ret, skip=MOM_SKIP, window=MOM_WINDOW)
    spread_expanded, spread_raw = build_spread(
        residual_ret, pairs=PAIRS, window=SPREAD_WINDOW,
        adf_pval=ADF_PVAL, halflife_max=HALFLIFE_MAX, verbose=False,
    )

    # Regime detection
    print("[plot] Fitting DCC-GARCH regime model ...")
    try:
        regime_df = compute_regime_signals(residual_ret[ASSETS], spread_raw)
    except Exception as exc:
        print(f"[plot] WARNING: DCC failed ({exc}). Defaulting to 'normal'.")
        regime_df = pd.DataFrame({"regime": "normal"}, index=residual_ret.index)

    # Weekly covariance (medium-term)
    print("[plot] Building covariance matrices (weekly) ...")
    returns_all = residual_ret[ASSETS]
    reb_dates   = returns_all.resample("W-FRI").last().index
    cov_dict    = OrderedDict()
    for date in reb_dates:
        win = returns_all.loc[:date].iloc[-COV_WINDOW:]
        if len(win) < COV_WINDOW:
            continue
        lw = LedoitWolf().fit(win.values)
        cov_dict[date] = pd.DataFrame(lw.covariance_, index=ASSETS, columns=ASSETS)

    # Daily covariance (short-term futures)
    print("[plot] Building covariance matrices (daily, short-term) ...")
    cov_dict_st = OrderedDict()
    for date in df[FUTURES].index:
        win = df[FUTURES].loc[:date].iloc[-ST_COV_WINDOW:]
        if len(win) < ST_COV_WINDOW:
            continue
        lw = LedoitWolf().fit(win.values)
        cov_dict_st[date] = pd.DataFrame(lw.covariance_, index=FUTURES, columns=FUTURES)

    # FM alignment
    print("[plot] FM alignment ...")
    reb_dates_list            = sorted(cov_dict.keys())
    alpha_mom_aligned,    _   = align_signal_to_expected_return(
        mom_final[ASSETS],       ret[ASSETS], reb_dates_list)
    alpha_spread_aligned, _   = align_signal_to_expected_return(
        spread_expanded[ASSETS], ret[ASSETS], reb_dates_list)
    alpha_mom_n    = normalize_alpha(alpha_mom_aligned.dropna(how="all"))
    alpha_spread_n = normalize_alpha(alpha_spread_aligned.dropna(how="all"))

    # Standalone backtests (for attribution)
    print("[plot] Running standalone Books ...")
    _book_params = dict(
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
    res_mom_sa    = Book("mom_sa",    alpha_mom_n,    **_book_params).run(ret[ASSETS])
    res_spread_sa = Book("spread_sa", alpha_spread_n, **_book_params).run(ret[ASSETS])

    # Rolling Sharpe combination (Mom + Spread)
    pnl_blocks = pd.DataFrame({
        "mom":    res_mom_sa["pnl"],
        "spread": res_spread_sa["pnl"],
    }).dropna(how="all")

    common_idx_a = alpha_mom_n.index.intersection(alpha_spread_n.index)
    EQ2          = pd.Series({"mom": 0.5, "spread": 0.5})
    w_rows       = {}
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
    alpha_combined = pd.DataFrame(combined_rows, index=ASSETS).T
    alpha_combined.index = pd.DatetimeIndex(alpha_combined.index)

    # Medium-term book
    print("[plot] Running medium-term and short-term Books ...")
    mt_book = Book("medium_term", alpha_combined, **_book_params)
    res_mt  = Allocator([mt_book]).run(ret[ASSETS])

    # Short-term reversal book (regime-gated)
    rev_raw_st   = -df[FUTURES].rolling(3).sum()
    alpha_rev_st = normalize_alpha(rev_raw_st)
    regime_sorted = regime_df.sort_index()
    scale_rows    = {}
    for date in alpha_rev_st.index:
        avail = regime_sorted.loc[:date]
        lbl   = avail["regime"].iloc[-1] if len(avail) > 0 else "normal"
        if not isinstance(lbl, str) or pd.isna(lbl):
            lbl = "normal"
        scale_rows[date] = get_book_actions(lbl)["short_term"]["alpha_multiplier"]
    alpha_rev_gated = alpha_rev_st.mul(pd.Series(scale_rows), axis=0)

    st_book = Book(
        "short_term", alpha_rev_gated, cov_dict_st, list(cov_dict_st.keys()),
        ST_GAMMA, ST_KAPPA, LAMBD, ST_MAX_WEIGHT, ST_TARGET_VOL, ST_EWMA_HL,
        SCALE_MIN, SCALE_MAX,
    )
    # Short-term book uses simple returns (not log), per run_engine.py comment 2026-03-31
    res_st = st_book.run(df[ASSETS])

    pnl_mt = res_mt["book_results"]["medium_term"]["pnl"]
    pnl_st = res_st["pnl"]

    # Rolling IC for signal quality plot
    print("[plot] Computing rolling IC ...")
    ic_mom    = _compute_rolling_ic(alpha_mom_n[ASSETS],    ret[ASSETS], reb_dates_list)
    ic_spread = _compute_rolling_ic(alpha_spread_n[ASSETS], ret[ASSETS], reb_dates_list)

    print("[plot] Pipeline complete.")
    return {
        "pnl_mt":        pnl_mt,
        "pnl_st":        pnl_st,
        "pnl_mom_sa":    res_mom_sa["pnl"],
        "pnl_spread_sa": res_spread_sa["pnl"],
        "ic_mom":        ic_mom,
        "ic_spread":     ic_spread,
    }


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_cumulative_pnl(
    pnl_mt:        pd.Series,
    pnl_st:        pd.Series,
    pnl_mom_sa:    pd.Series,
    pnl_spread_sa: pd.Series,
    out_dir:       Path,
) -> None:
    """Cumulative return: combined strategy + momentum and spread attribution."""
    pnl_combined = pnl_mt.add(pnl_st, fill_value=0.0).sort_index()

    def cum_pct(s: pd.Series) -> pd.Series:
        s = s.dropna().sort_index()
        return ((1 + s).cumprod() - 1) * 100   # percentage

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(cum_pct(pnl_combined),
            color=C_COMBINED, lw=1.8, label="Combined (MT + ST)", zorder=3)
    ax.plot(cum_pct(pnl_mt),
            color=C_MT_ONLY,  lw=1.0, ls="--", label="Medium-term only", zorder=2)
    ax.plot(cum_pct(pnl_mom_sa),
            color=C_MOMENTUM, lw=1.0, alpha=0.85, label="Momentum standalone", zorder=2)
    ax.plot(cum_pct(pnl_spread_sa),
            color=C_SPREAD,   lw=1.0, alpha=0.85, label="Spread standalone", zorder=2)
    ax.axhline(0, color="black", lw=0.6, ls="--", alpha=0.35)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return (%)")
    ax.set_title("Cumulative PnL — Cross-Asset Statistical Arbitrage Engine")
    ax.legend(loc="upper left")
    fig.tight_layout()
    _save(fig, out_dir / "cumulative_pnl.png")


def plot_drawdown(
    pnl_mt:  pd.Series,
    pnl_st:  pd.Series,
    out_dir: Path,
) -> None:
    """Drawdown curve for the combined strategy."""
    pnl  = pnl_mt.add(pnl_st, fill_value=0.0).dropna().sort_index()
    cum  = (1 + pnl).cumprod()
    peak = cum.cummax()
    dd   = ((cum - peak) / peak) * 100   # percentage, negative

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.fill_between(dd.index, dd.values, 0, color=C_DD, alpha=0.35, label="Drawdown")
    ax.plot(dd.index, dd.values, color=C_DD, lw=0.8, alpha=0.6)
    ax.axhline(0, color="black", lw=0.6, ls="--", alpha=0.35)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.set_title("Drawdown — Combined Strategy")
    ax.legend(loc="lower left")
    fig.tight_layout()
    _save(fig, out_dir / "drawdown.png")


def plot_rolling_sharpe(
    pnl_mt:  pd.Series,
    pnl_st:  pd.Series,
    out_dir: Path,
    window:  int = 52,
) -> None:
    """Rolling annualized Sharpe of the combined strategy."""
    pnl = pnl_mt.add(pnl_st, fill_value=0.0).dropna().sort_index()

    # Auto-detect annualization frequency from index density
    n_days      = max((pnl.index[-1] - pnl.index[0]).days, 1)
    obs_per_day = len(pnl) / n_days
    freq        = 252 if obs_per_day > 0.5 else 52

    roll_mean   = pnl.rolling(window, min_periods=window // 2).mean()
    roll_std    = pnl.rolling(window, min_periods=window // 2).std().clip(lower=1e-12)
    roll_sharpe = np.sqrt(freq) * roll_mean / roll_std

    mean_sharpe = roll_sharpe.mean()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(roll_sharpe.index, roll_sharpe.values,
            color=C_COMBINED, lw=1.5, label=f"{window}-period rolling Sharpe")
    ax.axhline(0,           color="black",     lw=0.8, ls="--", alpha=0.4)
    ax.axhline(mean_sharpe, color=C_COMBINED,  lw=0.8, ls=":",  alpha=0.7,
               label=f"Mean = {mean_sharpe:.2f}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Annualized Sharpe")
    ax.set_title(f"Rolling Sharpe — Combined Strategy  ({window}-period window)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    _save(fig, out_dir / "rolling_sharpe.png")


def plot_rolling_ic(
    ic_mom:    pd.Series,
    ic_spread: pd.Series,
    out_dir:   Path,
) -> None:
    """Rolling mean IC for Momentum and Spread signals (26-week window)."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(ic_mom.dropna().index,    ic_mom.dropna().values,
            color=C_MOMENTUM, lw=1.2, label="Momentum (26w rolling IC)")
    ax.plot(ic_spread.dropna().index, ic_spread.dropna().values,
            color=C_SPREAD,   lw=1.2, label="Spread (26w rolling IC)")
    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.4)
    ax.set_xlabel("Date")
    ax.set_ylabel("IC (Spearman rank correlation)")
    ax.set_title("Rolling IC — Momentum & Spread Signals  (26-week window)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    _save(fig, out_dir / "rolling_ic.png")


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved: {path.relative_to(ROOT)}")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[plot] Output directory: {FIG_DIR.relative_to(ROOT)}")

    results = _run_pipeline()

    plot_cumulative_pnl(
        pnl_mt        = results["pnl_mt"],
        pnl_st        = results["pnl_st"],
        pnl_mom_sa    = results["pnl_mom_sa"],
        pnl_spread_sa = results["pnl_spread_sa"],
        out_dir       = FIG_DIR,
    )
    plot_drawdown(
        pnl_mt  = results["pnl_mt"],
        pnl_st  = results["pnl_st"],
        out_dir = FIG_DIR,
    )
    plot_rolling_sharpe(
        pnl_mt  = results["pnl_mt"],
        pnl_st  = results["pnl_st"],
        out_dir = FIG_DIR,
    )
    plot_rolling_ic(
        ic_mom    = results["ic_mom"],
        ic_spread = results["ic_spread"],
        out_dir   = FIG_DIR,
    )

    print(f"\n[plot] All figures written to reports/figures/")
