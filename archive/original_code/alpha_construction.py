#!/usr/bin/env python
# coding: utf-8

# # This notebook is the second stage of the process. It loads the cleaned and unified dataset and builds the alphas we will use for optimization.

# Stage 0: Load, clean, calculate log-returns

# In[1]:


# Import packages

import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.regression.rolling import RollingOLS
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import pickle


# In[3]:


# Paths

cleaned_dataset_path = Path(r"C:\Users\pcarg\OneDrive\Υπολογιστής\MFE UCLA\Last Quarter\Classes\Statistical Arbitrage\Final Project\Data\unified_cleaned_dataset.csv")


# In[5]:


# Load dataset, set to datetime and set and sort index

df = pd.read_csv(cleaned_dataset_path)
df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date').sort_index()


# In[7]:


# Asset groups

futures = ['CLc1','Cc1','HGc1','LCOc1','NGc1','Wc1']
etfs    = ['JETS','XLB','XLE','XLI','XLP','XLU','XLY']
market  = ['SPX']

assets = futures + etfs
all_assets = assets + market   


# In[9]:


# Clean any stray NaNs first (these cause log1p warnings)

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(how='any')   


# In[11]:


# Ensure all values > -1 (log1p domain)

df = df.clip(lower=-0.999999)


# In[13]:


# Compute log returns

ret = np.log1p(df)


# In[15]:


# ret has same shape as df except possible dropped rows

print(ret.shape)


# Stage 1: Beta neutralize. Everything downstream must be built on beta neutralized excess returns to ensure truly idiosyncratic alpha.

# In[18]:


# Empty matrix for residual returns and regression parameters

residual_ret = pd.DataFrame(index=ret.index, columns=assets)
window = 60
mkt = ret['SPX']


# In[20]:


# Run rolling regression to neutralize

for asset in assets:
    y = ret[asset]
    X = sm.add_constant(mkt)

    rols = RollingOLS(y, X, window=window)
    res = rols.fit()

    beta_t  = res.params['SPX']
    alpha_t = res.params['const']

    residual_ret[asset] = y - (alpha_t + beta_t * mkt)


# In[21]:


# ---- Drop initial NaNs from rolling window ----

residual_ret = residual_ret.dropna()

print("Residual return matrix shape:", residual_ret.shape)
print(residual_ret.head())


# In[22]:


# Print shape and head

print("Residual return matrix shape:", residual_ret.shape)
print(residual_ret.head())


# Stage 2: Short term reversal, but this time not vanilla.

# In[27]:


# Helper function to cross sectionally z-score per day

def cs_normalize(signal_df):
    """Cross-sectional z-score normalization per day."""
    return (signal_df - signal_df.mean(axis=1, skipna=True).values.reshape(-1,1)) \
           / signal_df.std(axis=1, skipna=True).values.reshape(-1,1)


# In[29]:


# Helper function to winsorize robustly

def robust_winsorize(df, z=3.0):
    """MAD-based winsorization, fully vectorized and Pandas-safe."""
    med = df.median(axis=1)
    mad = (df.sub(med, axis=0)).abs().median(axis=1) + 1e-8
    
    # Compute z-scores with safe broadcasting
    z_scores = df.sub(med, axis=0).div(mad, axis=0)

    # Cap values using clip
    df_w = df.copy()
    df_w = df_w.clip(lower=med - z*mad, upper=med + z*mad, axis=0)

    return df_w


# In[31]:


# Compute short-term reversal from residual returns

rev_raw = - residual_ret.rolling(window=5).sum()


# In[33]:


# First pass normalization

rev_norm = cs_normalize(rev_raw)


# In[35]:


# Winsorize heavy tails

rev_wins = robust_winsorize(rev_norm, z=3.0)


# In[36]:


# Final normalization after winsorization

rev_final = rev_wins           


# In[39]:


# Inspect

print("Reversal signal shape:", rev_final.shape)
print(rev_final.tail())


# Stage 3: Medium term momentum revisited. Modifications since before: We build momentum on neutralized returns. We incorporate a skip period. Apply cross sectional normalization, winsorization 

# In[42]:


# Skip first 5 days of residual returns

residual_shifted = residual_ret.shift(5)


# In[44]:


# 60-day window (days 6 to 65 in raw returns)

mom_raw = residual_shifted.rolling(window=60).sum()


# In[46]:


# First normalization

mom_norm = cs_normalize(mom_raw)


# In[48]:


# Winsorize

mom_wins = robust_winsorize(mom_norm, z=3.0)


# In[50]:


# Final normalization

mom_final = mom_wins


# In[52]:


# Inspect

print("Momentum signal shape:", mom_final.shape)
print(mom_final.tail())


# Stage 4 (REMOVED): Carry removed.
# Dataset contains only front-month contracts (c1 suffix).
# Proper carry requires futures term structure (F1/F2 slope), which is unavailable.


# Stage 5: Cointegration signal.

# We construct a statistically grounded relative-value alpha by exploiting long-run
# equilibrium relationships between sector ETFs and the commodity futures that
# economically drive them. For each ETF–commodity pair, we estimate a rolling
# 120-day hedge ratio via OLS:
# 
# $$ 
# ETF_t = a_t + b_t * COM_t + u_t
# $$

# The spread is defined as the residual series u_t. To eliminate look-ahead bias, the ADF
# test is applied WITHIN each rolling 120-day window only — never on the full sample.
# At each time t, the signal is non-zero only if ADF p-value < 0.10 in that window.
# The spread is also normalized using rolling z-score statistics from the same window.

# For cointegrated windows (rolling ADF p < 0.10 at time t):
#
# $$
# alpha_{spread,t} = – zscore_rolling(u_t)
# $$
#
# (negative sign because a positive z-score implies ETF overvaluation relative
# to the commodity). Non-cointegrated windows receive alpha = 0.

# Cross-sectional normalization and MAD winsorization are then applied for comparability.

# In[74]:


# Define ETF–commodity pairs

pairs = [
    ('XLE',  'CLc1'),
    ('XLE',  'LCOc1'),
    ('XLE',  'NGc1'),
    ('XLI',  'HGc1'),
    ('XLB',  'HGc1'),
    ('XLB',  'Cc1'),
    ('XLB',  'Wc1'),
    ('XLY',  'Wc1'),
    ('JETS', 'CLc1'),
]


# In[76]:


# Initialize spread signal matrix, define rolling window for hedge ratio

spread_raw = pd.DataFrame(index=residual_ret.index,
                          columns=[f"{etf}_{com}" for etf, com in pairs])

window = 120  


# In[78]:


# Loop

for etf, com in pairs:
    
    # Extract residual returns for OLS input
    y = residual_ret[etf]
    X = residual_ret[com]
    
    # Rolling OLS: ETF_t = a + b * COM_t + u_t
    rols = RollingOLS(y, sm.add_constant(X), window=window).fit()
    alpha_t = rols.params['const']
    beta_t  = rols.params[com]
    
    # Residual spread = u_t
    spread = y - (alpha_t + beta_t * X)
    
    spread_arr = spread.values

    # Rolling ADF + rolling z-score (NO look-ahead bias).
    # For each t, use only spread[t-window+1 : t+1].
    signal = pd.Series(0.0, index=spread.index)

    for i in range(window - 1, len(spread_arr)):
        w_data  = spread_arr[i - window + 1 : i + 1]
        w_clean = w_data[~np.isnan(w_data)]

        if len(w_clean) < window // 2:
            continue

        # ADF test on rolling window only — eliminates look-ahead bias
        try:
            pval = adfuller(w_clean)[1]
        except Exception:
            pval = 1.0

        if pval >= 0.10:
            continue  # not cointegrated in this window → zero signal

        # Rolling z-score using window statistics only
        mu  = w_clean.mean()
        std = w_clean.std()
        if std < 1e-8:
            continue

        signal.iloc[i] = -(spread_arr[i] - mu) / std  # negative sign = mean reversion

    spread_raw[f"{etf}_{com}"] = signal


# In[79]:


# After building all spreads → normalize and winsorize

spread_norm = cs_normalize(spread_raw)
spread_wins = robust_winsorize(spread_norm, z=3.0)
spread_final = spread_wins


# In[80]:


# Inspect

print("Spread alpha shape:", spread_final.shape)
print(spread_final.tail())


# Stage 6: We align spread signals

# In[82]:


# Create empty matrix for ETF-aligned spreads

spread_expanded = pd.DataFrame(0.0, 
                               index=residual_ret.index, 
                               columns=residual_ret.columns)


# In[87]:


# Map each spread pair to its ETF

for etf, com in pairs:
    colname = f"{etf}_{com}"
    spread_expanded[etf] += spread_final[colname]

# Average spreads per ETF

for etf in etfs:
    # Count how many spreads contribute
    count = sum([1 for e,c in pairs if e == etf])
    if count > 1:
        spread_expanded[etf] /= count


# In[89]:


# Inspect

print("Expanded spread matrix shape:", spread_expanded.shape)
print(spread_expanded.tail())


# Stage 8: Risk parity weighting of alphas.

# In[92]:


# Compute volatility of each alpha component (3 blocks: reversal, momentum, spread)

vol_rev    = rev_final.std().mean()
vol_mom    = mom_final.std().mean()
vol_spread = spread_expanded.std().mean()


# In[94]:


# Inspect

print("Volatility of alpha blocks:")
print(f"  Reversal:     {vol_rev:.6f}")
print(f"  Momentum:     {vol_mom:.6f}")
print(f"  Spreads:      {vol_spread:.6f}")


# In[98]:


# Risk-parity weights: w_i ∝ 1 / vol_i

raw_w = np.array([1/vol_rev, 1/vol_mom, 1/vol_spread])
w = raw_w / raw_w.sum()


# In[100]:


# Store weights nicely:

weights = {
    "reversal": w[0],
    "momentum": w[1],
    "spreads":  w[2]
}


# In[102]:


# Inspect

print("\nRisk-Parity Weights:")
for k,v in weights.items():
    print(f"  {k:10s} → {v:.4f}")


# In[104]:


# Create common index to align from intersection of signals

common_index = (
    rev_final.index
    .intersection(mom_final.index)
    .intersection(spread_expanded.index)
)


# In[106]:


# Ensure columns are always identical across all assets

common_columns = rev_final.columns


# In[108]:


# Finalize respective alphas

alpha_rev    = rev_final.loc[common_index, common_columns]
alpha_mom    = mom_final.loc[common_index, common_columns]
alpha_spread = spread_expanded.loc[common_index, common_columns]


# In[110]:


# Combine alpha blocks using these weights

alpha_combined = (
      weights["reversal"] * rev_final
    + weights["momentum"] * mom_final
    + weights["spreads"]  * spread_expanded
)


# In[112]:


# Inspect

print("\nCombined alpha matrix shape:", alpha_combined.shape)


# At this point, because we want to backtest both the performance of individual alphas and of their combination. So we output all of them.

# In[125]:


alpha_mom.tail(10)


# In[123]:


alpha_rev.tail(10)


# In[121]:


alpha_combined.tail(10)


# In[119]:


# Save

alpha_rev.to_pickle("alpha_rev_no_bl.pkl")
alpha_mom.to_pickle("alpha_mom_no_bl.pkl")
alpha_spread.to_pickle("alpha_spread_no_bl.pkl")
alpha_combined.to_pickle("alpha_combined_no_bl.pkl")

print("Saved all alpha block files.")


# ---
