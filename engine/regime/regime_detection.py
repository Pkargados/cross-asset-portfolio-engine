#!/usr/bin/env python
# coding: utf-8
"""
regime_detection.py — DCC-GARCH regime detection for alpha scaling.

DCC is fit ONCE on the full sample:
  - Parameters (a, b) are full-sample estimates (minor look-ahead in params only).
  - R_t path is recursively filtered using those params — standard practice for
    regime conditioning ("Option A" confirmed by user 2026-03-28).

Public API:
    compute_regime_signals(returns_df, spread_raw_df, ...) -> pd.DataFrame
    compute_alpha_scale(regime_df, reb_dates) -> pd.Series
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

def _find_project_root() -> Path:
    """Walk up from this file until CLAUDE.md (project root marker) is found."""
    current = Path(__file__).resolve().parent
    for _ in range(10):
        if (current / "CLAUDE.md").exists():
            return current
        current = current.parent
    raise RuntimeError(
        f"Project root not found: CLAUDE.md not located within 10 levels of "
        f"{Path(__file__).resolve()}"
    )

sys.path.insert(0, str(_find_project_root() / "engine" / "risk" / "dcc_garch"))

from garch.gjr_garch import fit_multivariate_gjr
from dcc import fit as dcc_fit


# ============================================================================
# PUBLIC API
# ============================================================================

def compute_regime_signals(
    returns_df,
    spread_raw_df,
    ewma_lambda: float = 0.94,
    spike_window: int  = 60,
    spike_z: float     = 2.0,
) -> pd.DataFrame:
    """
    Compute daily regime signals from DCC-GARCH and EWMA volatility.

    Parameters
    ----------
    returns_df    : pd.DataFrame (T, N) — daily log returns, ASSETS columns.
                    Must have no NaN/Inf after dropna().
    spread_raw_df : pd.DataFrame (T, n_pairs) — raw pair spread signals.
                    Used to compute spread stability (fraction of active pairs).
    ewma_lambda   : float — EWMA decay factor for realized vol (default 0.94).
    spike_window  : int   — rolling window (days) for rho mean/std (default 60).
    spike_z       : float — z-score threshold for a correlation spike (default 2.0).

    Returns
    -------
    pd.DataFrame indexed by dates, with columns:
        rho_t         — mean off-diagonal DCC correlation at each day
        disp_t        — std of off-diagonal DCC correlations at each day
        spike_t       — 1.0 if rho_t > rho_roll_mean + spike_z * rho_roll_std
        ewma_vol_t    — cross-asset mean EWMA annualized vol
        spread_stab   — fraction of pairs with non-zero signal each day
        rho_roll_mean — rolling mean of rho_t (spike_window days)
        rho_roll_std  — rolling std  of rho_t (spike_window days)
    """
    # ── 1. Clean returns — DCC raises ValueError on NaN/Inf ──────────────────
    ret_clean  = returns_df.dropna()
    ret_matrix = ret_clean.values  # (T, N) ndarray

    if not np.all(np.isfinite(ret_matrix)):
        raise ValueError(
            "returns_df contains non-finite values after dropna(). "
            "Ensure ASSETS columns are fully populated before calling DCC."
        )

    T, N = ret_matrix.shape
    print(f"    [DCC] Fitting GJR-GARCH on {T} obs × {N} assets ...")

    # ── 2. GARCH stage: fit univariate GJR-GARCH per asset ───────────────────
    garch_result = fit_multivariate_gjr(ret_matrix)
    Z      = garch_result['Z']       # (T, N) standardized residuals
    sigmas = garch_result['sigmas']  # (T, N) % daily vol

    # ── 3. DCC stage: fit DCC on standardized residuals ──────────────────────
    print(f"    [DCC] Fitting DCC (model='DCC') ...")
    dcc_result = dcc_fit(Z, sigmas, model='DCC')
    R_t = dcc_result['R']   # (T, N, N) conditional correlation path

    a, b = dcc_result['params']
    print(
        f"    [DCC] {'Converged' if dcc_result['converged'] else 'WARNING: did not converge'}: "
        f"a={a:.4f}  b={b:.4f}  llh={dcc_result['llh']:.1f}"
    )

    # ── 4. Off-diagonal correlation statistics ────────────────────────────────
    off_diag_mask = ~np.eye(N, dtype=bool)  # True for all off-diagonal entries
    rho_arr  = np.array([R_t[t][off_diag_mask].mean() for t in range(T)])
    disp_arr = np.array([R_t[t][off_diag_mask].std()  for t in range(T)])

    dates      = ret_clean.index
    rho_series = pd.Series(rho_arr, index=dates)

    # ── 5. Rolling stats for spike detection ─────────────────────────────────
    rho_roll_mean = rho_series.rolling(spike_window, min_periods=spike_window // 2).mean()
    rho_roll_std  = rho_series.rolling(spike_window, min_periods=spike_window // 2).std()

    # rho_z: standardised deviation of current correlation from its rolling mean.
    # Positive = correlation above its recent baseline (risk-on crowding or crisis).
    rho_z   = (rho_series - rho_roll_mean) / rho_roll_std.clip(lower=1e-8)
    spike_t = (rho_z > spike_z).astype(float)

    # ── 5b. Rolling z-score for dispersion ───────────────────────────────────
    # disp_t measures cross-asset heterogeneity: high dispersion means assets are
    # moving differently despite elevated average correlation (sector rotation /
    # clustered stress).  Low dispersion means uniform co-movement (crowded exits).
    disp_series    = pd.Series(disp_arr, index=dates)
    disp_roll_mean = disp_series.rolling(spike_window, min_periods=spike_window // 2).mean()
    disp_roll_std  = disp_series.rolling(spike_window, min_periods=spike_window // 2).std()
    disp_z         = (disp_series - disp_roll_mean) / disp_roll_std.clip(lower=1e-8)

    # ── 6. EWMA cross-asset mean annualized volatility ────────────────────────
    # EWMA variance recursion: var_t = (1-λ)·r²_t + λ·var_{t-1}
    sq_ret   = ret_matrix ** 2                   # (T, N)
    ewma_var = np.empty_like(sq_ret)
    ewma_var[0] = sq_ret[0]
    for t in range(1, T):
        ewma_var[t] = (1.0 - ewma_lambda) * sq_ret[t] + ewma_lambda * ewma_var[t - 1]
    # Cross-asset mean vol, annualized (daily → annual: sqrt(252))
    ewma_vol_t = pd.Series(
        np.sqrt(ewma_var.mean(axis=1)) * np.sqrt(252),
        index=dates,
    )

    # ── 7. Spread stability: fraction of pairs with non-zero signal ───────────
    spread_aligned = spread_raw_df.reindex(dates)
    spread_stab    = (spread_aligned.abs() > 1e-8).mean(axis=1)

    # ── 8. Categorical regime label ───────────────────────────────────────────
    # Conditions evaluated in priority order (np.select picks the first True).
    # All inputs are backward-looking at each t — no look-ahead bias.
    #
    #   "broken"    — spread_stab < 0.3: fewer than 30% of pairs are active.
    #                 Signal infrastructure is too sparse to trust; regime label
    #                 flags this for downstream filtering, not for scaling.
    #
    #   "crisis"    — rho_z > 2.0: correlation has spiked > 2σ above its rolling
    #                 baseline. Classic flight-to-safety / de-leveraging event.
    #                 High systematic risk; mean-reversion signals unreliable.
    #
    #   "crowded"   — rho_z ∈ (1,2] AND disp_z < 0: elevated correlation but
    #                 dispersion is BELOW its own rolling mean. Assets are moving
    #                 together homogeneously — crowded positioning, uniform exits.
    #
    #   "clustered" — rho_z ∈ (1,2] AND disp_z ≥ 0: elevated correlation but
    #                 dispersion is AT OR ABOVE its rolling mean. Some asset groups
    #                 are correlated within clusters but diverge across clusters —
    #                 sector rotation or partial stress.
    #
    #   "normal"    — all other dates: correlation within its historical baseline.
    regime_labels = pd.Series(
        np.select(
            condlist=[
                spread_stab < 0.3,                      # broken
                rho_z > 2.0,                            # crisis
                (rho_z > 1.0) & (disp_z < 0),          # crowded
                (rho_z > 1.0) & (disp_z >= 0),         # clustered
            ],
            choicelist=["broken", "crisis", "crowded", "clustered"],
            default="normal",
        ),
        index=dates,
        dtype=object,
    )

    # ── 9. Assemble output DataFrame ─────────────────────────────────────────
    regime_df = pd.DataFrame({
        "rho_t":         rho_series,
        "disp_t":        disp_series,
        "spike_t":       spike_t,
        "ewma_vol_t":    ewma_vol_t,
        "spread_stab":   spread_stab,
        "rho_roll_mean": rho_roll_mean,
        "rho_roll_std":  rho_roll_std,
        "rho_z":         rho_z,
        "disp_z":        disp_z,
        "regime":        regime_labels,
    })

    n_spikes = int(spike_t.sum())
    print(
        f"    [DCC] Regime signals computed: shape={regime_df.shape}  "
        f"avg_rho={rho_arr.mean():.3f}  disp_mean={disp_arr.mean():.3f}  "
        f"n_spikes={n_spikes} ({100.0*n_spikes/T:.1f}% of days)"
    )
    return regime_df


# LEGACY — deprecated as of 2026-03-30.
# This function implements regime as a scalar multiplier on the combined alpha,
# which is structurally incorrect: vol targeting re-inflates positions after the
# scalar is applied, partially cancelling the regime guard.  Additionally, a
# single scalar cannot distinguish between regime states that require different
# portfolio shapes (e.g. boost spreads in "clustered" vs. disable them in "broken").
# Use regime/regime_mapping.get_book_actions() in the new engine instead.
# Retained here for backward compatibility with run_baseline.py ONLY.
def compute_alpha_scale(
    regime_df,
    reb_dates,
) -> pd.Series:
    """
    Compute multiplicative alpha scale in [0.1, 1.0] per rebalancing date.

    Uses regime signal from the most recent day <= rebalancing date (no look-ahead).

    Scale logic:
        scale = 1.0
        if spike_t == 1:  scale *= 0.5
        rho_z = (rho_t - rho_roll_mean) / rho_roll_std
        if rho_z > 1.0:   scale *= max(0.3, 1.0 - 0.35*(rho_z - 1.0))
        return clip(scale, 0.1, 1.0)

    Parameters
    ----------
    regime_df : pd.DataFrame — output of compute_regime_signals
    reb_dates : DatetimeIndex — rebalancing Friday dates

    Returns
    -------
    pd.Series indexed by reb_dates, values in [0.1, 1.0]
    """
    scale_vals      = {}
    regime_sorted   = regime_df.sort_index()

    for date in sorted(reb_dates):
        # Most recent regime observation on or before this rebalancing date
        avail = regime_sorted.loc[:date]
        if len(avail) == 0:
            scale_vals[date] = 1.0
            continue

        row   = avail.iloc[-1]
        scale = 1.0

        # Spike: flatten positions by 50% on correlation crisis days
        if row["spike_t"] == 1.0:
            scale *= 0.5

        # Elevated correlation: smooth linear taper above 1 z-score
        rho_std = float(row["rho_roll_std"]) if pd.notna(row["rho_roll_std"]) else 0.0
        if rho_std > 1e-8:
            rho_z = (float(row["rho_t"]) - float(row["rho_roll_mean"])) / rho_std
        else:
            rho_z = 0.0

        if rho_z > 1.0:
            scale *= max(0.3, 1.0 - 0.35 * (rho_z - 1.0))

        scale_vals[date] = float(np.clip(scale, 0.1, 1.0))

    scale_series = pd.Series(scale_vals)
    n_adjusted   = int((scale_series < 1.0).sum())
    print(
        f"    [Regime] Alpha scale: mean={scale_series.mean():.3f}  "
        f"min={scale_series.min():.3f}  max={scale_series.max():.3f}  "
        f"n_scaled_down={n_adjusted}/{len(scale_series)}"
    )
    return scale_series
