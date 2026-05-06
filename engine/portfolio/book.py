#!/usr/bin/env python
# coding: utf-8
"""
portfolio/book.py — Book abstraction.

A Book owns a single independent portfolio: alpha, covariance, optimizer
parameters, and vol-targeting configuration.  It is regime-unaware; the
Allocator layer decides whether a Book is active and at what weight.

Public API
----------
    Book(name, alpha_df, cov_dict, reb_dates, gamma, kappa, lambd,
         max_weight, target_vol, ewma_halflife, scale_min, scale_max,
         is_active=True)

    book.run(returns_df) -> dict
        Executes the full backtest loop.
        Returns: {"weights", "pnl", "sharpe", "max_dd",
                  "turnover", "avg_scale", "n_cap_bind"}

Implementation note
-------------------
Book.run() is a verbatim extraction of run_baseline.py::run_backtest
(Phase 3, Step 4).  Zero logic changes.  The reference implementation
in run_baseline.py is NOT deleted — it remains as the regression baseline.
"""

import numpy as np
import pandas as pd
from collections import OrderedDict

from engine.portfolio.optimizer import chernov_weights


class Book:
    """
    Independent portfolio unit.

    Parameters
    ----------
    name          : str                 — identifier ("medium_term", etc.)
    alpha_df      : pd.DataFrame (T×N) — alpha in E[r] units (FM-aligned)
    cov_dict      : OrderedDict        — {date: pd.DataFrame} LW Σ per reb date
    reb_dates     : list or DatetimeIndex — rebalancing schedule
    gamma         : float              — risk-aversion (Chernov optimizer)
    kappa         : float              — position inertia
    lambd         : float              — L1 transaction-cost penalty
    max_weight    : float              — per-asset position limit (abs value)
    target_vol    : float              — annualized vol target
    ewma_halflife : int                — EWMA halflife (weeks) for realized vol
    scale_min     : float              — floor on vol-targeting scale factor
    scale_max     : float              — ceiling on vol-targeting scale factor
    is_active     : bool               — toggled by Allocator / regime layer
    """

    def __init__(
        self,
        name: str,
        alpha_df: pd.DataFrame,
        cov_dict: OrderedDict,
        reb_dates,
        gamma: float,
        kappa: float,
        lambd: float,
        max_weight: float,
        target_vol: float,
        ewma_halflife: int,
        scale_min: float,
        scale_max: float,
        is_active: bool = True,
    ):
        self.name          = name
        self.alpha_df      = alpha_df
        self.cov_dict      = cov_dict
        self.reb_dates     = list(reb_dates)
        self.gamma         = gamma
        self.kappa         = kappa
        self.lambd         = lambd
        self.max_weight    = max_weight
        self.target_vol    = target_vol
        self.ewma_halflife = ewma_halflife
        self.scale_min     = scale_min
        self.scale_max     = scale_max
        self.is_active     = is_active

    # ------------------------------------------------------------------
    # run() — verbatim extraction of run_baseline.py::run_backtest
    # ------------------------------------------------------------------

    def run(self, returns_df: pd.DataFrame) -> dict:
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

        EWMA initialized at rv ≈ TARGET_VOL (neutral prior → scale = 1 at t=0).
        Ledoit-Wolf Σ used only for relative sizing; NOT for portfolio vol estimation.

        Returns dict with weights, pnl, sharpe, max_dd, turnover, avg_scale, n_cap_bind.
        """
        # ── unpack self attributes (mirrors run_backtest parameter names) ──────
        alpha_df      = self.alpha_df
        cov_dict      = self.cov_dict
        gamma         = self.gamma
        kappa         = self.kappa
        lambd         = self.lambd
        max_weight    = self.max_weight
        target_vol    = self.target_vol
        ewma_halflife = self.ewma_halflife
        scale_min     = self.scale_min
        scale_max     = self.scale_max

        # ── verbatim from run_backtest (run_baseline.py lines 147–264) ─────────
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
            # ── update EWMA vol with last week's realized PnL ─────────────────
            # At date d_i, the return for (d_{i-1}, d_i] has just been observed.
            # Update ewma_var BEFORE sizing new positions — no look-ahead.
            if prev_x is not None and prev_date in weekly_ret_map:
                pnl_t  = float(np.dot(prev_x, weekly_ret_map[prev_date]))
                ewma_var = (1.0 - ewma_alpha) * ewma_var + ewma_alpha * pnl_t ** 2

            alpha_t = alpha_df.loc[date, assets]
            Sigma_t = cov_dict[date].loc[assets, assets]

            # ── 1. Optimize (Chernov + Ledoit-Wolf Σ, unchanged) ─────────────
            x_t = chernov_weights(alpha_t, Sigma_t, x_prev, n,
                                  gamma, kappa, lambd, max_weight)

            # ── 2. EWMA power scaling (cap-aware) ─────────────────────────────
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
