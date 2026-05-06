#!/usr/bin/env python
# coding: utf-8

# # This notebook is just a check on the Refinitiv data Jade sent

# In[4]:


# Import packages

import pandas as pd
from pathlib import Path


# In[103]:


# Path

futures_path = Path(r"C:\Users\pcarg\OneDrive\Υπολογιστής\MFE UCLA\Last Quarter\Classes\Statistical Arbitrage\Final Project\Data\commodity_futures.csv")
etf_path = Path(r"C:\Users\pcarg\OneDrive\Υπολογιστής\MFE UCLA\Last Quarter\Classes\Statistical Arbitrage\Final Project\Data\sector_etf.csv")
market_path = Path(r"C:\Users\pcarg\OneDrive\Υπολογιστής\MFE UCLA\Last Quarter\Classes\Statistical Arbitrage\Final Project\Data\spx.csv")
output_path = Path(r"C:\Users\pcarg\OneDrive\Υπολογιστής\MFE UCLA\Last Quarter\Classes\Statistical Arbitrage\Final Project\Data\unified_cleaned_dataset.csv")


# In[8]:


# Load data

df_fut = pd.read_csv(futures_path)
df_etf = pd.read_csv(etf_path)
df_mkt = pd.read_csv(market_path)


# In[10]:


# Inspect

df_fut.head(), df_etf.head(), df_mkt.head()
df_fut.info(), df_etf.info(), df_mkt.info()


# In[14]:


# Convert dates to datetime, sort, drop dupes

for df in [df_fut, df_etf, df_mkt]:
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    df.drop_duplicates(subset=['Date'], inplace=True)


# In[18]:


# One more check for dates

df_fut['Date'].is_monotonic_increasing


# In[20]:


# Check missingness

df_fut.isna().mean().sort_values()
df_etf.isna().mean().sort_values()
df_mkt.isna().mean().sort_values()


# In[22]:


# Check futures columns

df_fut.columns


# In[24]:


# Check ETF columns

df_etf.columns


# In[28]:


# Check dates alignment between datasets

dates_fut = set(df_fut['Date'])
dates_etf = set(df_etf['Date'])
dates_mkt = set(df_mkt['Date'])

common_dates = sorted(list(dates_fut & dates_etf & dates_mkt))
len(common_dates), len(dates_fut), len(dates_etf), len(dates_mkt)


# ETFs are the limiting calendar, and we will align the entire dataset to their trading dates.

# In[30]:


# Restrict dataset on common dates

common = pd.DataFrame({'date': common_dates})


# In[34]:


# Monotonicity check to find jumps

df_fut.set_index('Date').pct_change().describe(percentiles=[.01, .05, .95, .99])


# Cc1 is front-month contract of corn. Min is -0.173, max is +0.08. Somewhat tame returns but maybe realistic for an agricultural commodity. We will pay attention going forward. 

# NGc1 extreme moves are kind of anticipated, and Wc1 huge move is probably during COVID. Continuous series likely rolled before the jump. We will winsorize at 3$\sigma$ before creating alphas to ensure this works.

# In[36]:


# Check instrument count - dimensionality

num_fut = df_fut.shape[1] - 1
num_etf = df_etf.shape[1] - 1
num_assets = num_fut + num_etf
num_assets


# 13 assets is a reasonable size. Would be better if it were a bit more, but that's ok.

# In[40]:


# Outlier - bad data point check

(df_fut.set_index('Date').pct_change().abs() > 0.3).sum().sort_values()


# 1 and 3 outliers for NG (natural gas) and CL (WTI crude) is completely ok. Commodities experience extreme days. We move on.

# In[50]:


# Build unified price panel

df_all = (
    df_fut.merge(df_etf, on="Date", how="inner")
          .merge(df_mkt[['Date','Price']], on="Date", how="inner")
)


# In[52]:


# Sort, reset index

df_all = df_all.sort_values('Date').reset_index(drop=True)


# In[54]:


# Compute simple returns

df_ret = df_all.set_index('Date').pct_change().dropna()


# We get simple and not log returns because alphas will require z-score of raw returns.

# In[59]:


# Inspect unified data file

df_all.head()


# In[61]:


df_all.tail()


# In[63]:


df_all.columns


# In[65]:


df_ret.head()


# In[67]:


df_etf[['XLB']].head(10)


# Issue: ETF data is in daily percentage returns. Let's go back.

# In[84]:


# Get ETF returns in decimals

df_etf_ret = df_etf.copy()
df_etf_ret.iloc[:,1:] /= 100


# In[86]:


# Get normal returns for futures + SPX

df_fut_ret = df_fut.set_index("Date").pct_change()
df_spx_ret = df_mkt.set_index("Date")["Price"].pct_change()


# In[88]:


# Align on common dates, cleanup

df_all_ret = (
    df_fut_ret.merge(df_etf_ret.set_index("Date"), left_index=True, right_index=True, how="inner")
              .merge(df_spx_ret.rename("SPX"), left_index=True, right_index=True, how="inner")
)
df_all_ret = df_all_ret.dropna()


# In[90]:


df_all_ret.head(10)


# In[92]:


df_all_ret.tail(10)


# In[98]:


# Make sure Date is datetime index
df = df_all_ret.copy()
df.index = pd.to_datetime(df.index)

# Add a year column
df["Year"] = df.index.year

# Store results here
rows = []

for col in df.columns.drop("Year"):
    stats = df.groupby("Year")[col].agg(
        mean_return = "mean",
        var_return  = "var",
        min_return  = "min",
        max_return  = "max"
    )
    # add asset name
    stats["Asset"] = col
    rows.append(stats.reset_index())

# Combine
annual_stats = pd.concat(rows, ignore_index=True)

# Order columns nicely
annual_stats = annual_stats[["Asset", "Year", "mean_return", "var_return", "min_return", "max_return"]]

annual_stats


# Reasonable numbers. Save dataset for next step.

# In[105]:


# Save

df_all_ret.to_csv(output_path)


# ---
