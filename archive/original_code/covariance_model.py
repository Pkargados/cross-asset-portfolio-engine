#!/usr/bin/env python
# coding: utf-8

# # This notebook calculated the covariance matrix necessary for the optimization of the portfolio using Ledoit Wolf shrinkage.

# In this section we construct a rolling, weekly-updated covariance matrix
# for our universe of 14 assets (commodity futures, sector ETFs, SPX).
# 
# We use:
# - Daily returns (`df_all_ret`)
# - 60 trading-day rolling window
# - Ledoit–Wolf shrinkage estimator (robust, PD, stable)
# - Weekly rebalancing schedule (Fridays)

# In[44]:


# Import packages

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from collections import OrderedDict
from pathlib import Path
import pickle


# In[3]:


# Path

cleaned_dataset_path = Path(r"C:\Users\pcarg\OneDrive\Υπολογιστής\MFE UCLA\Last Quarter\Classes\Statistical Arbitrage\Final Project\Data\unified_cleaned_dataset.csv")


# In[11]:


# Load

returns = pd.read_csv(cleaned_dataset_path)


# In[13]:


# Convert datetime, set index, print shape

returns['Date'] = pd.to_datetime(returns['Date'])
returns = returns.set_index('Date')
print(returns.shape)
returns.head()


# We rebalance weekly (Friday), so we want a covariance matrix updated
# on each weekly date.
# 
# Using the weekly frequency:
# - reduces noise
# - stabilizes weights
# - aligns with medium-frequency stat arb
# 

# In[16]:


# Covariance window length

window = 60


# In[18]:


# Weekly rebalancing dates (Fridays)

rebalance_freq = 'W-FRI'
rebalance_dates = returns.resample(rebalance_freq).last().index


# In[20]:


# Inspect

print("Num rebalance dates:", len(rebalance_dates))
print("First few:", rebalance_dates[:5])


# This makes sense, given about 10.5 years of rebalancing Fridays. Now we compute the rolling Ledoit Wolf covariances.

# For each rebalance date t:
# 1. Take the last 60 daily returns up to t
# 2. Fit Ledoit–Wolf shrinkage estimator
# 3. Store $\Sigma_{t}$ in a dictionary

# We skip dates where we have fewer than 60 observations.

# In[25]:


covs = OrderedDict()

for date in rebalance_dates:
    
    # 1. Extract return window (last 60 observations before or on 'date')
    window_data = returns.loc[:date].iloc[-window:]
    
    if len(window_data) < window:
        # Not enough data at the beginning of the sample
        continue
    
    # 2. Fit Ledoit–Wolf
    lw = LedoitWolf().fit(window_data.values)
    cov = lw.covariance_
    
    # 3. Store
    covs[date] = cov

len(covs)


# 540 makes sense as a number of matrices generated.

# We convert each Σ_t to a pandas DataFrame for readability 
# and for easier portfolio optimization later
# 

# In[33]:


# Build assets list

assets = returns.columns.tolist()
print(len(assets))
print(assets)


# In[35]:


# Convert to dataframe based on assets list

cov_matrices = {
    date: pd.DataFrame(covs[date], index=assets, columns=assets)
    for date in covs
}


# In[37]:


# Inspect covariance matrix for first available date

first_date = next(iter(cov_matrices))
cov_matrices[first_date]


# We validate:
# - No NaN values in covariance matrices
# - They are positive definite
# - Shape is 14x14 (consistent with asset count)
# 

# In[39]:


# Check shape, NaNs

for date in list(cov_matrices.keys())[:3]:
    print(date, cov_matrices[date].shape)

nan_counts = {
    date: cov_matrices[date].isna().sum().sum()
    for date in cov_matrices
}
max(nan_counts.values())


# Above is: No NaNs, symmetric, futures vol seems higher than ETF vol, XLU and XLP are low-vol being defensive, related assets (energy, oil futures) reasonably correlated.

# In[41]:


# Check positive definiteness (smallest eigenvalue > 0)

test_date = list(cov_matrices.keys())[10]  # arbitrary mid-sample date
eigvals = np.linalg.eigvals(cov_matrices[test_date])
print("Min eigenvalue:", eigvals.min())


# Healthy. Save and proceed.

# In[49]:


# Save

with open("cov_matrices.pkl", "wb") as f:
    pickle.dump(cov_matrices, f)

print("Saved cov_matrices.pkl")


# In[55]:


# Also save returns to pickle

returns.to_pickle("df_all_ret.pkl")
print("Saved df_all_ret.pkl")


# ---

# In[61]:


# Daily Ledoit–Wolf covariance matrices for Black Litterman

print("Building DAILY covariance matrices (for Black–Litterman)...")

window = 60
daily_covs = OrderedDict()
dates = returns.index

for date in dates:
    
    window_data = returns.loc[:date].iloc[-window:]
    
    if len(window_data) < window:
        continue
    
    lw = LedoitWolf().fit(window_data.values)
    cov = lw.covariance_
    
    daily_covs[date] = pd.DataFrame(cov,
                                    index=assets,
                                    columns=assets)

print("Daily covariance matrices:", len(daily_covs))

# Save daily matrices

with open("cov_matrices_daily.pkl", "wb") as f:
    pickle.dump(daily_covs, f)

print("Saved cov_matrices_daily.pkl")


# ---
