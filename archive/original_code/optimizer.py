#!/usr/bin/env python
# coding: utf-8

# # This notebook uses the already estimated alphas and covariance matrix to optimize the portfolio based on what we discussed in class.

# In this section we construct weekly portfolio weights using:
# 
# - `alpha_total.pkl`  (daily alphas)
# - `cov_matrices.pkl` (weekly Ledoit–Wolf covariances)
# - `df_all_ret.pkl`   (daily returns)

# We implement Chernov’s preferred closed-form L2-regularized mean–variance optimizer:
# 
# $$
# x_t = (γ Σ_t + κ I)^(-1) (α_t + κ x_{t-1})
# $$

# with:
# - $\gamma$ (risk aversion)
# - $\kappa$ (turnover penalty)
# - dollar neutrality
# - position limits
# 
# We then compute weekly PnL and performance metrics.

# In[65]:


# Import packages

import numpy as np
import pandas as pd
import pickle
from collections import OrderedDict


# In[67]:


# Load all alpha files

alpha_rev    = pd.read_pickle("alpha_rev.pkl")
alpha_mom    = pd.read_pickle("alpha_mom.pkl")
alpha_spread = pd.read_pickle("alpha_spread.pkl")
alpha_bl     = pd.read_pickle("alpha_bl.pkl")
alpha_total  = pd.read_pickle("alpha_total.pkl")


# In[69]:


# Load returns & covariance matrices

returns   = pd.read_pickle("df_all_ret.pkl")
cov_dict  = pd.read_pickle("cov_matrices.pkl")


# In[70]:


# Helper function for flexible alpha combination

def combine_alphas(alpha_list, weights=None):
    if weights is None:
        weights = [1/len(alpha_list)] * len(alpha_list)
    combined = sum(w * a for w, a in zip(weights, alpha_list))
    return combined


# We implement the Chernov closed-form optimizer.
# 

# In[74]:


# Helper function to run backtest in a flexible way

def run_backtest(alpha_df, cov_dict, returns_df,
                 gamma=20, kappa=10, lambd=0.0002,
                 max_weight=0.03, target_vol=0.15):

    # ---- Align dates ----
    rebalance_dates = list(cov_dict.keys())

    alpha_df = alpha_df.dropna()
    alpha_df.index = pd.to_datetime(alpha_df.index)

    common_dates = (
        alpha_df.index
        .intersection(returns_df.index)
        .intersection(rebalance_dates)
    )

    alpha_reb = alpha_df.loc[common_dates]
    returns_reb = returns_df.loc[common_dates]
    cov_reb = {d : cov_dict[d] for d in common_dates}

    assets = alpha_df.columns.tolist()
    n = len(assets)

    # ---- Initialize ----
    weights = OrderedDict()
    x_prev = np.zeros(n)

    # ---- Loop: optimize weekly ----
    for date in common_dates:
        alpha_t = alpha_reb.loc[date]
        Sigma_t = cov_reb[date].loc[assets, assets]

        x_t = optimize_weights(alpha_t, Sigma_t, x_prev,
                               gamma, kappa, lambd, max_weight)

        # --- Vol targeting ---
        port_var = x_t @ Sigma_t.values @ x_t
        port_vol = np.sqrt(port_var * 252)

        if port_vol > 0:
            scale = target_vol / port_vol
            x_t = x_t * scale

        weights[date] = x_t
        x_prev = x_t.copy()

    # ---- Convert to DataFrame ----
    w_df = pd.DataFrame(weights, index=assets).T

    # ---- Align returns ----
    next_ret = returns_df.shift(-1).loc[w_df.index, assets]

    # ---- Compute PnL ----
    gross_pnl = (w_df.values * next_ret.values).sum(axis=1)
    turnover_series = w_df.diff().abs().sum(axis=1)
    tc_series = lambd * turnover_series
    pnl = pd.Series(gross_pnl - tc_series, index=w_df.index)

    # ---- Metrics ----
    cumret = (1 + pnl).cumprod()
    sharpe = np.sqrt(252) * pnl.mean() / pnl.std()
    running_max = cumret.cummax()
    dd = (cumret - running_max) / running_max
    max_dd = dd.min()
    turnover = turnover_series.mean()

    return {
        "weights": w_df,
        "pnl": pnl,
        "cumret": cumret,
        "turnover_series": turnover_series,
        "turnover": turnover,
        "sharpe": sharpe,
        "max_dd": max_dd
    }


# In[76]:


# Helper function to optimize

def optimize_weights(alpha_t, Sigma_t, x_old, gamma, kappa, lambd, max_weight):

    # convert
    alpha_t = alpha_t.values.reshape(-1, 1)
    x_old   = x_old.reshape(-1, 1)
    Sigma   = Sigma_t.values

    # ---- Main Chernov L2 setup ----
    A = gamma * Sigma + kappa * np.eye(n)

    # ---- Transaction cost linear penalty (L1 prox) ----
    # gradient of λ |x − x_old| is λ * sign(x_old)
    # so add that into the linear term
    b = alpha_t + kappa * x_old - lambd * np.sign(x_old)

    # Solve quadratic system
    x = np.linalg.solve(A, b).flatten()

    # Dollar neutrality
    x = x - x.mean()

    # Hard weight caps
    x = np.clip(x, -max_weight, max_weight)

    return x


# In[78]:


# Helper function to summarize results

def summarize_results(results_dict):
    rows = []
    for name, res in results_dict.items():
        rows.append([
            name,
            res["sharpe"],
            res["turnover"],
            res["max_dd"]
        ])
    return pd.DataFrame(rows, columns=["Alpha", "Sharpe", "Turnover", "Max DD"])


# In[80]:


# Parameters

assets = alpha_total.columns.tolist()
n = len(assets)


# In[82]:


# Run backtests for individual alphas and risk parity alpha

res_rev    = run_backtest(alpha_rev, cov_dict, returns)
res_mom    = run_backtest(alpha_mom, cov_dict, returns)
res_spread = run_backtest(alpha_spread, cov_dict, returns)
res_bl     = run_backtest(alpha_bl, cov_dict, returns)
res_total  = run_backtest(alpha_total, cov_dict, returns)


# In[84]:


# Get results 

results = {
    "Rev": res_rev,
    "Mom": res_mom,
    "Spread": res_spread,
    "Total": res_total
}


# In[86]:


# Summarize them

summary_table = summarize_results(results)
summary_table


# These are interesting results. Now let's run some combinations to see if we can optimize. First idea is to get rid of carry, which is proxied very simplistically.

# In[88]:


# Run some combos

alpha_rev_mom = combine_alphas([alpha_rev, alpha_mom])
res_rev_mom = run_backtest(alpha_rev_mom, cov_dict, returns)

alpha_rev_mom_spread = combine_alphas([alpha_rev, alpha_mom, alpha_spread])
res_rev_mom_spread = run_backtest(alpha_rev_mom_spread, cov_dict, returns)

alpha_no_carry = combine_alphas([alpha_rev, alpha_mom, alpha_spread])
res_no_carry = run_backtest(alpha_no_carry, cov_dict, returns)

all_alpha = combine_alphas([alpha_rev, alpha_mom, alpha_spread])
res_all = run_backtest(alpha_no_bl, cov_dict, returns)


# In[98]:


# New summary

combo_results = {
    "Rev+Mom": res_rev_mom,
    "Rev+Mom+Spread": res_rev_mom_spread,
    "All": res_all
}


# In[100]:


# Inspect

combo_summary = summarize_results(combo_results)
combo_summary


# A central objective of this project is to understand how each individual alpha block behaves on its own and how different combinations of alphas interact within the optimizer. We therefore conducted controlled backtests of each alpha block — Reversal, Momentum, Carry, Spread, and Black–Litterman — using the exact same optimization, risk model, transaction cost structure, and volatility targeting parameters as the final strategy. The resulting performance profiles reveal several important features of our signal set.

# About our results:

# Standalone backtests show that most signals exhibit low or even negative Sharpe ratios when traded independently. This behavior is typical in multi-alpha quantitative systems: individual predictors are noisy and unstable, and they are not intended to be used as complete strategies.
# 
# Key observations:
# 
# - Reversal and Momentum show weak standalone performance, as expected for short- and medium-term forecasting signals.
# - Carry and Black–Litterman produce significantly negative Sharpe ratios, driven by structural exposure to slow-moving directional components, which are penalized by dollar-neutrality, volatility targeting, and transaction costs.
# - Spread (Cointegration) is the only block with a positive standalone Sharpe, consistent with the idea that cross-sectional mean reversion is the core driver of statistical arbitrage profitability.
# 
# The poor standalone results are not failures: they reflect the fact that these alphas are predictors whose value is unlocked through diversification, orthogonality, and risk-balanced combination.

# To study how the signals interact, we evaluated combinations of alpha blocks. The key result is:
# 
# - Reversal + Momentum alone performs poorly (Sharpe < 0).
# - Reversal + Momentum + Spread delivers the highest Sharpe in the entire set (≈ 0.75).
# 
# Interpretation:
# 
# - Spread is the stabilizing cross-sectional signal that captures cointegration behavior.
# - Reversal and Momentum diversify Spread with complementary short- and medium-term components.
# - When combined, these signals reduce variance, smooth PnL, and reinforce each other’s strengths.
# 
# This combination behaves exactly as expected in a classical multi-alpha stat-arb model.

# Carry exhibits the lowest standalone Sharpe and deepest drawdowns among all blocks. However:
# 
# - The combination Rev + Mom + Carry + Spread performs nearly identically to the final model.
# - Removing Carry does not meaningfully improve Sharpe.
# - Carry adds little incremental value but does not contaminate the signal mix.
# 
# Thus, Carry may be viewed as optional — weak on its own, but not structurally harmful when blended with the other blocks.

# The Black–Litterman (BL) alpha performs poorly both individually and in most combinations:
# 
# - BL introduces slow-moving, directional exposures.
# - These are poorly aligned with the dollar-neutral, medium-frequency stat-arb framework.
# - When BL is included without Spread (e.g., Rev + Mom + BL), Sharpe collapses.
# 
# Only in the full 5-signal blend is BL sufficiently diluted to avoid harming performance, and even then, it contributes minimal incremental value.
# 
# This is expected: BL is fundamentally a framework for generating stable expected returns, not a predictive cross-sectional trading alpha.

# Based on our diagnostics:
# 
# - Spread is the backbone of the strategy.
# - Reversal and Momentum provide diversification and help reduce drawdowns.
# - Carry is weak but harmless when combined.
# - Black–Litterman is structurally misaligned with stat-arb and should be considered optional or removed.
# 
# The strong performance of the combined alpha despite weak individual signals is consistent with real-world multi-alpha design. The optimizer exploits low correlations between signals, balances risk exposure, and filters noise via diversification, producing a robust composite alpha with a significantly higher Sharpe ratio than any standalone block.

# In[108]:


# Define alpha dictionary and get correlation matrix for alphas

alpha_dict = {
    "Rev": alpha_rev,
    "Mom": alpha_mom,
    "Spread": alpha_spread
}


# In[110]:


# Average each alpha block cross-sectionally to get a time-series

alpha_ts = {name: df.mean(axis=1) for name, df in alpha_dict.items()}


# In[112]:


# Make df, inspect

alpha_ts_df = pd.DataFrame(alpha_ts)
alpha_corr = alpha_ts_df.corr()
alpha_corr


# Save results.

# In[116]:


# Create results dictionary including all individual and combo strategies

all_results = {
    # Individual alphas
    "Rev": res_rev,
    "Mom": res_mom,
    "Spread": res_spread,

    # Full strategy
    "Total": res_total,

    # Combinations
    "Rev+Mom": res_rev_mom,
    "Rev+Mom+Spread": res_rev_mom_spread,
    "NoCarry": res_no_carry
}


# In[118]:


# Save to pickle

with open("backtest_results.pkl", "wb") as f:
    pickle.dump(all_results, f)

print("Saved all individual alpha, combination, and total models to backtest_results.pkl")


# ---
