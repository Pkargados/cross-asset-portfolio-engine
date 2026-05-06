#!/usr/bin/env python
# coding: utf-8
"""
regime/regime_mapping.py — Regime-to-portfolio decision policy.

This module defines the control layer between the regime state and portfolio
construction.  It is NOT a model — it encodes economic hypotheses about how
each regime should alter alpha usage per book.

All mappings are initial priors and MUST be validated via conditional
IC / rolling Sharpe analysis before being treated as reliable.  Do not
optimize these parameters on backtest Sharpe — that would introduce
look-ahead bias in the policy itself.

────────────────────────────────────────────────────────────────
REGIME DEFINITIONS (from regime_detection.py)
────────────────────────────────────────────────────────────────

  "normal"
      rho_z ≤ 1.0, spread_stab ≥ 0.3
      Cross-asset correlation is within its historical baseline.
      All signal types are operating in their intended environment.

  "clustered"
      rho_z ∈ (1.0, 2.0], disp_z ≥ 0, spread_stab ≥ 0.3
      Correlation is elevated BUT cross-asset dispersion is also above its
      own baseline.  Some asset groups are highly correlated within clusters
      (e.g. energy ETF + crude futures) while others diverge.  This is a
      sector-rotation or partial-stress environment — not a systemic crisis.

  "crowded"
      rho_z ∈ (1.0, 2.0], disp_z < 0, spread_stab ≥ 0.3
      Correlation is elevated AND dispersion is BELOW its baseline.  Assets
      are moving together homogeneously — a signature of crowded positioning
      and uniform de-leveraging across the book.

  "crisis"
      rho_z > 2.0, spread_stab ≥ 0.3
      Correlation has spiked more than 2σ above its rolling mean.
      Classic flight-to-safety / forced liquidation event.  All mean-reversion
      premises are fragile; the primary goal is capital preservation.

  "broken"
      spread_stab < 0.3 (takes priority over all rho conditions)
      Fewer than 30% of cointegrated pairs are currently active (ADF + ECM
      filter gates are failing).  The spread signal infrastructure is too
      sparse to trade.  This is an infrastructure signal, not a market signal.

────────────────────────────────────────────────────────────────
REGIME → BOOK DECISION TABLE  (initial hypotheses — not tuned)
────────────────────────────────────────────────────────────────

  Regime       │ spread            │ momentum          │ short_term
  ─────────────┼───────────────────┼───────────────────┼──────────────────
  normal       │ active  × 1.0     │ active  × 1.0     │ active  × 1.0
  clustered    │ active  × 1.3     │ active  × 1.0     │ active  × 0.8
  crowded      │ active  × 0.6     │ active  × 1.0     │ active  × 0.8
  crisis       │ active  × 0.4     │ active  × 0.5     │ active  × 0.3
  broken       │ DISABLED× 0.0     │ active  × 1.0     │ active  × 1.0

ECONOMIC RATIONALE PER REGIME
──────────────────────────────

  normal:
    All books at full weight.  No regime adjustment warranted.

  clustered (elevated rho, high disp_z → sector rotation):
    - Spread: BOOST (×1.3).  When assets diverge within and across clusters,
      cointegrated pairs are more likely to be pulled apart and revert.
      The spread signal has more edge in a cluster-rotation environment.
    - Momentum: unchanged.  Medium-term trend is not disrupted by cluster moves.
    - Short-term reversal: slight reduction (×0.8).  Intra-day noise increases
      during sector rotation, reducing short-horizon mean-reversion reliability.

  crowded (elevated rho, low disp_z → uniform co-movement):
    - Spread: REDUCE (×0.6).  When all assets move together, cointegrated pairs
      lose their idiosyncratic component — spread signals are driven by noise,
      not genuine mean-reversion.  High risk of false signals.
    - Momentum: unchanged (×1.0).  Momentum benefits from directional moves;
      uniform co-movement does not disrupt a diversified momentum signal.
    - Short-term reversal: slight reduction (×0.8).  Crowded exits create
      momentum at short horizons, working against mean-reversion.

  crisis (rho_z > 2σ → systemic stress):
    - Spread: strong reduction (×0.4).  During crises, cointegration breaks
      down as fundamentals become irrelevant.  Forced liquidation drives spreads
      away from equilibrium for extended periods.
    - Momentum: moderate reduction (×0.5).  Momentum can work in crises
      (persistent downtrends) but with much higher variance.  Retain some exposure.
    - Short-term reversal: heavy reduction (×0.3).  Crisis reversals are
      unreliable — gaps and limit moves make mean-reversion hazardous.

  broken (spread_stab < 0.3 → infrastructure failure):
    - Spread: DISABLE (active=False, ×0.0).  The ADF + ECM filter is rejecting
      >70% of windows.  There is no spread signal to trade, not a market signal
      about spread quality.  Disabling is correct — trading on sparse signal
      would amplify noise.
    - Momentum and short-term: unchanged.  These signals do not depend on
      cointegration infrastructure.

NOTE ON alpha_multiplier:
  The alpha_multiplier is applied to the signal BEFORE the optimizer.
  This is different from post-hoc position scaling:
    x_t = (γΣ + κI)^-1 (α_t * multiplier + κx_{t-1})
  Changing α_t changes the optimizer's solution shape (relative weights
  across assets), not just the overall position size.  It is a more
  economically meaningful intervention than scalar post-scaling.

────────────────────────────────────────────────────────────────
Public API:
    get_book_actions(regime_label: str) -> dict
    get_actions_for_date(regime_df: pd.DataFrame, date) -> dict
────────────────────────────────────────────────────────────────
"""

import pandas as pd


# ── Valid regime labels (for input validation) ───────────────────────────────
_VALID_REGIMES = {"normal", "clustered", "crowded", "crisis", "broken"}


def get_book_actions(regime_label: str) -> dict:
    """
    Map a categorical regime label to per-book portfolio decisions.

    This is a pure function: no state, no side effects, no data access.
    It encodes economic hypotheses — not empirically fitted parameters.

    Parameters
    ----------
    regime_label : str
        One of: "normal", "clustered", "crowded", "crisis", "broken".
        An unrecognised label falls through to the "normal" fallback with
        a warning rather than raising — this keeps the engine running if
        regime detection produces an unexpected value.

    Returns
    -------
    dict with keys "spread", "momentum", "short_term".
    Each value is a sub-dict:
        {
            "active":           bool   — if False, book produces zero weights
            "alpha_multiplier": float  — scale applied to alpha BEFORE optimizer
        }

    Usage example
    -------------
    actions = get_book_actions("crowded")
    # actions["spread"]["alpha_multiplier"] → 0.6
    # actions["spread"]["active"]           → True
    """

    if regime_label == "normal":
        # Baseline — no adjustment.
        return {
            "spread":     {"active": True,  "alpha_multiplier": 1.0},
            "momentum":   {"active": True,  "alpha_multiplier": 1.0},
            "short_term": {"active": True,  "alpha_multiplier": 1.0},
        }

    elif regime_label == "clustered":
        # Elevated correlation with high dispersion → sector rotation.
        # Boost spreads: pairs diverge more along cluster boundaries.
        # Reduce short-term: noise increases during sector re-allocation.
        return {
            "spread":     {"active": True,  "alpha_multiplier": 1.3},
            "momentum":   {"active": True,  "alpha_multiplier": 1.0},
            "short_term": {"active": True,  "alpha_multiplier": 0.8},
        }

    elif regime_label == "crowded":
        # Elevated correlation with low dispersion → homogeneous co-movement.
        # Reduce spreads: idiosyncratic component compressed, signals are noisy.
        # Reduce short-term: crowded exits create momentum, hurting mean-reversion.
        return {
            "spread":     {"active": True,  "alpha_multiplier": 0.6},
            "momentum":   {"active": True,  "alpha_multiplier": 1.0},
            "short_term": {"active": True,  "alpha_multiplier": 0.8},
        }

    elif regime_label == "crisis":
        # rho_z > 2σ → systemic stress / forced liquidation.
        # Reduce all books. Spread most severely (cointegration breaks down).
        # Retain some momentum (directional crises can have persistent trends).
        return {
            "spread":     {"active": True,  "alpha_multiplier": 0.4},
            "momentum":   {"active": True,  "alpha_multiplier": 0.5},
            "short_term": {"active": True,  "alpha_multiplier": 0.3},
        }

    elif regime_label == "broken":
        # spread_stab < 0.3 → ADF/ECM gates are rejecting >70% of windows.
        # There is no spread signal available — disabling is correct.
        # Momentum and short-term are unaffected (no dependence on cointegration).
        return {
            "spread":     {"active": False, "alpha_multiplier": 0.0},
            "momentum":   {"active": True,  "alpha_multiplier": 1.0},
            "short_term": {"active": True,  "alpha_multiplier": 1.0},
        }

    else:
        # Unknown label — fall back to normal rather than crashing.
        # This can occur if regime_detection produces NaN during warmup.
        import warnings
        warnings.warn(
            f"get_book_actions: unrecognised regime_label={repr(regime_label)}. "
            "Falling back to 'normal'. Check regime_detection output.",
            stacklevel=2,
        )
        return {
            "spread":     {"active": True,  "alpha_multiplier": 1.0},
            "momentum":   {"active": True,  "alpha_multiplier": 1.0},
            "short_term": {"active": True,  "alpha_multiplier": 1.0},
        }


def get_actions_for_date(regime_df: pd.DataFrame, date) -> dict:
    """
    Look up the most recent regime observation on or before `date` and
    return the corresponding book actions.

    No look-ahead bias: only rows with index ≤ date are considered.
    If no observation exists before `date` (e.g. during DCC warmup),
    falls back to "normal" actions silently.

    Parameters
    ----------
    regime_df : pd.DataFrame
        Output of compute_regime_signals().  Must contain a "regime" column
        with string labels.  Index must be a DatetimeIndex (daily).
    date : date-like
        The rebalancing date for which actions are needed.  Typically a
        Friday from the weekly rebalancing schedule.

    Returns
    -------
    dict — same structure as get_book_actions().

    Usage example
    -------------
    actions = get_actions_for_date(regime_df, pd.Timestamp("2020-03-20"))
    # During COVID crash → "crisis"
    # actions["spread"]["alpha_multiplier"] → 0.4
    """
    available = regime_df.loc[:date]

    if len(available) == 0:
        # No regime history yet — DCC warmup period.  Use neutral defaults.
        return get_book_actions("normal")

    regime_label = available["regime"].iloc[-1]

    # Guard against NaN that can appear at the start of the rolling window
    # (rho_z is undefined until spike_window observations have accumulated).
    if not isinstance(regime_label, str) or pd.isna(regime_label):
        return get_book_actions("normal")

    return get_book_actions(regime_label)
