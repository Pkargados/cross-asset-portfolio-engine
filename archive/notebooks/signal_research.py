#!/usr/bin/env python
# coding: utf-8

# ### This notebook is used as a playground to run statistical tests across signals to incorporate in my multi frequency stat arb trading engine 

# In[166]:


# Import

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import sys
sys.path.append(r"C:\Users\pcarg\OneDrive\Υπολογιστής\MFE UCLA\Last Quarter\Classes\Statistical Arbitrage\Final Project\Statistical Arbitrage Project")
from regime.regime_detection import compute_regime_signals
from regime.regime_mapping import get_book_actions
import statsmodels.api as sm
import matplotlib.pyplot as plt


# In[168]:


# Set style for plots

plt.style.use("default")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 12


# In[5]:


# Read

data = pd.read_csv("unified_cleaned_dataset.csv")


# In[7]:


data.head()


# In[9]:


data.tail()


# In[13]:


# Ensure datetime index

data['Date'] = pd.to_datetime(data['Date'])
data = data.set_index('Date')


# In[15]:


# Asset class definitions

commodities = ['CLc1','Cc1','HGc1','LCOc1','NGc1','Wc1']
etfs       = ['JETS','XLB','XLE','XLI','XLP','XLU','XLY']
market_col = 'SPX'


# In[17]:


# Split data

returns = data[commodities + etfs]
market = data[market_col]


# Raw Lehman reversal

# In[20]:


# Cross-sectional mean

cs_mean = returns.mean(axis=1)


# In[22]:


# Raw reversal signal

rev_raw = -(returns.sub(cs_mean, axis=0))


# In[24]:


# Z-score normalization

rev_raw_z = (rev_raw.sub(rev_raw.mean(axis=1), axis=0)
                      .div(rev_raw.std(axis=1), axis=0))


# Beta neutral reversal

# In[27]:


window = 60

residuals = pd.DataFrame(index=returns.index, columns=returns.columns)

for asset in returns.columns:
    df = pd.concat([returns[asset], market], axis=1).dropna()
    df.columns = ['r', 'mkt']
    
    cov = df['r'].rolling(window).cov(df['mkt'])
    var = df['mkt'].rolling(window).var()
    
    beta = cov / var
    res = df['r'] - beta * df['mkt']
    
    residuals.loc[res.index, asset] = res


# In[29]:


# Reversal on residuals

K = 3
rev_beta = -residuals.rolling(K).sum()


# In[31]:


# Normalize

rev_beta_z = (rev_beta.sub(rev_beta.mean(axis=1), axis=0)
                        .div(rev_beta.std(axis=1), axis=0))


# Future returns

# In[34]:


fwd_returns = returns.shift(-1)


# IC Functions

# In[97]:


def compute_ic(signal, future_returns):
    ic_list = []
    dates = []
    
    for t in signal.index[:-1]:
        x = signal.loc[t]
        y = future_returns.loc[t]
        
        mask = x.notna() & y.notna()
        
        if mask.sum() > 5:
            ic = spearmanr(x[mask], y[mask])[0]
            ic_list.append(ic)
            dates.append(t)
    
    return pd.Series(ic_list, index=dates)


def compute_ic_subset(signal, future_returns, asset_subset):
    ic_list = []
    dates = []
    
    for t in signal.index[:-1]:
        x = signal.loc[t, asset_subset]
        y = future_returns.loc[t, asset_subset]
        
        mask = x.notna() & y.notna()
        
        if mask.sum() > 3:
            ic = spearmanr(x[mask], y[mask])[0]
            ic_list.append(ic)
            dates.append(t)
    
    return pd.Series(ic_list, index=dates)


def compute_ic_decay(signal, returns, horizons=[1,2,3,5,10]):
    results = {}
    
    for h in horizons:
        fwd = returns.shift(-h)
        ic = compute_ic(signal, fwd)
        results[h] = ic.mean()
    
    return pd.Series(results)


def compute_ic_decay_subset(signal, returns, asset_subset, horizons=[1,2,3,5,10]):
    results = {}
    
    for h in horizons:
        fwd = returns.shift(-h)
        ic = compute_ic_subset(signal, fwd, asset_subset)
        results[h] = ic.mean()
    
    return pd.Series(results)


def summarize(name, ic):
    ic = ic.dropna()

    if len(ic) < 50:
        print(f"\n{name} (INSUFFICIENT DATA)")
        print("N:", len(ic))
        return

    X = np.ones(len(ic))

    model = sm.OLS(ic.values, X).fit(
        cov_type='HAC',
        cov_kwds={'maxlags': 5}
    )

    print(f"\n{name}")
    print("-"*50)
    print("Mean IC:", ic.mean())
    print("Std IC:", ic.std())
    print("NW t-stat:", model.tvalues[0])
    print("N:", len(ic))


# Run full IC

# In[100]:


# FULL UNIVERSE
ic_raw_full  = compute_ic(rev_raw_z, fwd_returns)
ic_beta_full = compute_ic(rev_beta_z, fwd_returns)

# BY ASSET CLASS
ic_raw_comm  = compute_ic_subset(rev_raw_z, fwd_returns, commodities)
ic_raw_etf   = compute_ic_subset(rev_raw_z, fwd_returns, etfs)

ic_beta_comm = compute_ic_subset(rev_beta_z, fwd_returns, commodities)
ic_beta_etf  = compute_ic_subset(rev_beta_z, fwd_returns, etfs)


# Print results

# In[103]:


# FULL
summarize("RAW - FULL", ic_raw_full)
summarize("BETA - FULL", ic_beta_full)

# BY GROUP
summarize("RAW - COMMODITIES", ic_raw_comm)
summarize("RAW - ETFs", ic_raw_etf)

summarize("BETA - COMMODITIES", ic_beta_comm)
summarize("BETA - ETFs", ic_beta_etf)


# IC Decay

# In[106]:


print("\nIC DECAY - RAW FULL")
print(compute_ic_decay(rev_raw_z, returns))

print("\nIC DECAY - BETA FULL")
print(compute_ic_decay(rev_beta_z, returns))

print("\nIC DECAY - RAW ETFs")
print(compute_ic_decay_subset(rev_raw_z, returns, etfs))

print("\nIC DECAY - RAW COMMODITIES")
print(compute_ic_decay_subset(rev_raw_z, returns, commodities))

print("\nIC DECAY - BETA ETFs")
print(compute_ic_decay_subset(rev_beta_z, returns, etfs))

print("\nIC DECAY - BETA COMMODITIES")
print(compute_ic_decay_subset(rev_beta_z, returns, commodities))


# Portfolio Check

# In[48]:


def compute_portfolio_returns(signal, returns):
    weights = signal.div(signal.abs().sum(axis=1), axis=0)
    pnl = (weights * returns.shift(-1)).sum(axis=1)
    return pnl

pnl_raw = compute_portfolio_returns(rev_raw_z, returns)
pnl_beta = compute_portfolio_returns(rev_beta_z, returns)

print("\nPortfolio RAW Sharpe:", pnl_raw.mean() / pnl_raw.std())
print("Portfolio BETA Sharpe:", pnl_beta.mean() / pnl_beta.std())


# In[51]:


# IC decay for commodity futures

print("\nIC DECAY - RAW COMMODITIES")
print(compute_ic_decay_subset(rev_raw_z, returns, commodities))

print("\nIC DECAY - BETA COMMODITIES")
print(compute_ic_decay_subset(rev_beta_z, returns, commodities))


# In[107]:


# IC decay for ETFs

print("\nIC DECAY - RAW ETFs")
print(compute_ic_decay_subset(rev_raw_z, returns, etfs))

print("\nIC DECAY - BETA ETFs")
print(compute_ic_decay_subset(rev_beta_z, returns, etfs))


# Compute regimes using our regime detection - labeling package

# In[65]:


# Package works with logreturns because it is DCC Garch based

log_returns = np.log(1 + returns)


# Let's see what this refers to.

# In[67]:


print(returns.min().sort_values())


# In[69]:


bad_rows = returns[returns['CLc1'] < -1]
bad_rows


# This is actually a real moment. It refers to crude prices turning negative during COVID. So we should not drop the rows for signal research as they have actual economic meaning. We should just remove it when we are trying to identify regime because garch will explode.

# In[73]:


returns_regime = returns.copy()

# clip extreme tails (e.g. 5-sigma or hard cap)
returns_regime = returns_regime.clip(lower=-0.5, upper=0.5)


# In[75]:


# Package works with logreturns because it is DCC Garch based

log_returns = np.log(1 + returns_regime)


# In[77]:


# Call function to detect regimes

regime_df = compute_regime_signals(
    returns_df=log_returns,
    spread_raw_df=pd.DataFrame(index=returns.index)
)


# In[79]:


# Isolate column as new series

regime_series = regime_df["regime"]


# In[87]:


def compute_ic_by_regime(signal, future_returns, regime_series, asset_subset=None):
    results = {}

    for regime in regime_series.dropna().unique():
        dates = regime_series[regime_series == regime].index
        
        ic_list = []

        for t in dates:
            if t not in signal.index or t not in future_returns.index:
                continue
            
            if asset_subset is None:
                x = signal.loc[t]
                y = future_returns.loc[t]
            else:
                x = signal.loc[t, asset_subset]
                y = future_returns.loc[t, asset_subset]

            mask = x.notna() & y.notna()

            if mask.sum() > 3:
                ic = spearmanr(x[mask], y[mask])[0]
                if not np.isnan(ic):
                    ic_list.append(ic)

        # stricter sample requirement (important)
        if len(ic_list) > 50:
            ic_series = pd.Series(ic_list)

            # Constant-only regression
            X = np.ones(len(ic_series))

            model = sm.OLS(ic_series.values, X).fit(
                cov_type='HAC',
                cov_kwds={'maxlags': 5}
            )

            results[regime] = {
                "mean_ic": ic_series.mean(),
                "nw_tstat": model.tvalues[0],   # Newey–West t-stat
                "std_ic": ic_series.std(),
                "n": len(ic_series)
            }

    return pd.DataFrame(results).T.sort_values("mean_ic", ascending=False)


# In[110]:


regime_series = regime_df["regime"]

print("BETA REVERSAL - FULL")
print(compute_ic_by_regime(rev_beta_z, fwd_returns, regime_series))

print("RAW REVERSAL - FULL")
print(compute_ic_by_regime(rev_raw_z, fwd_returns, regime_series))

print("BETA REVERSAL - COMMODITIES")
print(compute_ic_by_regime(rev_beta_z, fwd_returns, regime_series, commodities))

print("RAW REVERSAL - COMMODITIES")
print(compute_ic_by_regime(rev_raw_z, fwd_returns, regime_series, commodities))

print("BETA REVERSAL - ETFs")
print(compute_ic_by_regime(rev_beta_z, fwd_returns, regime_series, etfs))

print("RAW REVERSAL - ETFs")
print(compute_ic_by_regime(rev_raw_z, fwd_returns, regime_series, etfs))


# Now let's investigate IC decay by regime

# In[127]:


# Helper function 

def compute_ic_decay_by_regime(
    signal,
    returns,
    regime_series,
    asset_subset=None,
    horizons=[1,2,3,5,10]
):
    results = {}

    for regime in regime_series.dropna().unique():
        dates = regime_series[regime_series == regime].index

        horizon_results = {}

        for h in horizons:
            ic_list = []
            fwd = returns.shift(-h)

            for t in dates:
                if t not in signal.index or t not in fwd.index:
                    continue

                if asset_subset is None:
                    x = signal.loc[t]
                    y = fwd.loc[t]
                else:
                    x = signal.loc[t, asset_subset]
                    y = fwd.loc[t, asset_subset]

                mask = x.notna() & y.notna()

                if mask.sum() > 3:
                    ic = spearmanr(x[mask], y[mask])[0]
                    if not np.isnan(ic):
                        ic_list.append(ic)

            if len(ic_list) > 50:
                ic_series = pd.Series(ic_list)

                X = np.ones(len(ic_series))
                model = sm.OLS(ic_series.values, X).fit(
                    cov_type='HAC',
                    cov_kwds={'maxlags': max(5, h)}
                )

                horizon_results[h] = {
                    "mean_ic": ic_series.mean(),
                    "nw_tstat": model.tvalues[0],
                    "std_ic": ic_series.std(),
                    "n": len(ic_series)
                }

        if len(horizon_results) > 0:
            results[regime] = pd.DataFrame(horizon_results).T

    return results


# In[129]:


res_raw_full = compute_ic_decay_by_regime(rev_raw_z, returns, regime_series)
res_beta_full = compute_ic_decay_by_regime(rev_beta_z, returns, regime_series)

res_raw_comm = compute_ic_decay_by_regime(
    rev_raw_z, returns, regime_series, commodities
)

res_beta_comm = compute_ic_decay_by_regime(
    rev_beta_z, returns, regime_series, commodities
)

res_raw_etf = compute_ic_decay_by_regime(
    rev_raw_z, returns, regime_series, etfs
)

res_beta_etf = compute_ic_decay_by_regime(
    rev_beta_z, returns, regime_series, etfs
)


# In[130]:


def print_results(title, results):
    print(f"\n{'='*60}")
    print(title)
    print(f"{'='*60}")
    
    for regime, df in results.items():
        print(f"\n--- {regime.upper()} ---")
        print(df)


# In[135]:


print_results("RAW FULL", res_raw_full)
print_results("BETA FULL", res_beta_full)

print_results("RAW COMMODITIES", res_raw_comm)
print_results("BETA COMMODITIES", res_beta_comm)

print_results("RAW ETFs", res_raw_etf)
print_results("BETA ETFs", res_beta_etf)


# In[137]:


def pivot_mean_ic(results):
    out = {}
    for regime, df in results.items():
        out[regime] = df["mean_ic"]
    return pd.DataFrame(out)


# In[139]:


print("\nRAW — FULL UNIVERSE (MEAN IC)")
print(pivot_mean_ic(res_raw_full))

print("\nBETA — FULL UNIVERSE (MEAN IC)")
print(pivot_mean_ic(res_beta_full))

print("\nRAW — COMMODITIES (MEAN IC)")
print(pivot_mean_ic(res_raw_comm))

print("\nBETA — COMMODITIES (MEAN IC)")
print(pivot_mean_ic(res_beta_comm))

print("\nRAW — ETFs (MEAN IC)")
print(pivot_mean_ic(res_raw_etf))

print("\nBETA — ETFs (MEAN IC)")
print(pivot_mean_ic(res_beta_etf))


# Time split test. Question: Did something change post 2020? Is the relationship real or is it an artefact of our sample?

# In[142]:


split_date = "2020-01-01"


# In[155]:


def compute_ic_decay_by_regime_subset(
    signal,
    returns,
    regime_series,
    start_date=None,
    end_date=None,
    asset_subset=None,
    horizons=[1,2,3,5,10]
):
    mask = pd.Series(True, index=signal.index)

    if start_date is not None:
        mask &= signal.index >= pd.to_datetime(start_date)
    if end_date is not None:
        mask &= signal.index < pd.to_datetime(end_date)

    return compute_ic_decay_by_regime(
        signal.loc[mask],
        returns.loc[mask],
        regime_series.loc[mask],
        asset_subset=asset_subset,
        horizons=horizons
    )


# In[157]:


res_raw_full_pre = compute_ic_decay_by_regime_subset(
    rev_raw_z, returns, regime_series, end_date=split_date
)

res_raw_full_post = compute_ic_decay_by_regime_subset(
    rev_raw_z, returns, regime_series, start_date=split_date
)

res_beta_full_pre = compute_ic_decay_by_regime_subset(
    rev_beta_z, returns, regime_series, end_date=split_date
)

res_beta_full_post = compute_ic_decay_by_regime_subset(
    rev_beta_z, returns, regime_series, start_date=split_date
)

res_raw_comm_pre = compute_ic_decay_by_regime_subset(
    rev_raw_z, returns, regime_series, end_date=split_date, asset_subset=commodities
)

res_raw_comm_post = compute_ic_decay_by_regime_subset(
    rev_raw_z, returns, regime_series, start_date=split_date, asset_subset=commodities
)

res_beta_comm_pre = compute_ic_decay_by_regime_subset(
    rev_beta_z, returns, regime_series, end_date=split_date, asset_subset=commodities
)

res_beta_comm_post = compute_ic_decay_by_regime_subset(
    rev_beta_z, returns, regime_series, start_date=split_date, asset_subset=commodities
)

res_raw_etf_pre = compute_ic_decay_by_regime_subset(
    rev_raw_z, returns, regime_series, end_date=split_date, asset_subset=etfs
)

res_raw_etf_post = compute_ic_decay_by_regime_subset(
    rev_raw_z, returns, regime_series, start_date=split_date, asset_subset=etfs
)

res_beta_etf_pre = compute_ic_decay_by_regime_subset(
    rev_beta_z, returns, regime_series, end_date=split_date, asset_subset=etfs
)

res_beta_etf_post = compute_ic_decay_by_regime_subset(
    rev_beta_z, returns, regime_series, start_date=split_date, asset_subset=etfs
)


# In[159]:


print("\nRAW FULL PRE")
print(pivot_mean_ic(res_raw_full_pre))

print("\nRAW FULL POST")
print(pivot_mean_ic(res_raw_full_post))

print("\nBETA FULL PRE")
print(pivot_mean_ic(res_beta_full_pre))

print("\nBETA FULL POST")
print(pivot_mean_ic(res_beta_full_post))


print("\nRAW COMM PRE")
print(pivot_mean_ic(res_raw_comm_pre))

print("\nRAW COMM POST")
print(pivot_mean_ic(res_raw_comm_post))

print("\nBETA COMM PRE")
print(pivot_mean_ic(res_beta_comm_pre))

print("\nBETA COMM POST")
print(pivot_mean_ic(res_beta_comm_post))


print("\nRAW ETF PRE")
print(pivot_mean_ic(res_raw_etf_pre))

print("\nRAW ETF POST")
print(pivot_mean_ic(res_raw_etf_post))

print("\nBETA ETF PRE")
print(pivot_mean_ic(res_beta_etf_pre))

print("\nBETA ETF POST")
print(pivot_mean_ic(res_beta_etf_post))


# Now evaluate rolling IC.

# In[162]:


def compute_rolling_ic(signal, returns, window=252, asset_subset=None):
    fwd = returns.shift(-1)

    ic_list = []

    for t in signal.index[:-1]:
        if asset_subset is None:
            x = signal.loc[t]
            y = fwd.loc[t]
        else:
            x = signal.loc[t, asset_subset]
            y = fwd.loc[t, asset_subset]

        mask = x.notna() & y.notna()

        if mask.sum() > 3:
            ic = spearmanr(x[mask], y[mask])[0]
        else:
            ic = np.nan

        ic_list.append(ic)

    ic_series = pd.Series(ic_list, index=signal.index[:-1])
    return ic_series.rolling(window).mean()


# In[164]:


# FULL
roll_raw_full  = compute_rolling_ic(rev_raw_z, returns)
roll_beta_full = compute_rolling_ic(rev_beta_z, returns)

# COMMODITIES
roll_raw_comm  = compute_rolling_ic(rev_raw_z, returns, asset_subset=commodities)
roll_beta_comm = compute_rolling_ic(rev_beta_z, returns, asset_subset=commodities)

# ETFs
roll_raw_etf   = compute_rolling_ic(rev_raw_z, returns, asset_subset=etfs)
roll_beta_etf  = compute_rolling_ic(rev_beta_z, returns, asset_subset=etfs)


# Visualize a bit

# In[171]:


# Helper to plot heatmap for IC

def plot_ic_heatmap(results, title):
    df = pivot_mean_ic(results).sort_index()

    fig, ax = plt.subplots()

    im = ax.imshow(df.values, aspect='auto')

    ax.set_xticks(np.arange(len(df.columns)))
    ax.set_yticks(np.arange(len(df.index)))

    ax.set_xticklabels(df.columns)
    ax.set_yticklabels(df.index)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            val = df.iloc[i, j]
            ax.text(j, i, f"{val:.3f}", ha="center", va="center")

    ax.set_title(title)

    fig.colorbar(im)
    plt.show()


# In[173]:


# Call and plot

plot_ic_heatmap(res_raw_full,  "RAW Reversal — Full Universe")
plot_ic_heatmap(res_beta_full, "BETA Reversal — Full Universe")

plot_ic_heatmap(res_raw_comm,  "RAW Reversal — Commodities")
plot_ic_heatmap(res_beta_comm, "BETA Reversal — Commodities")

plot_ic_heatmap(res_raw_etf,   "RAW Reversal — ETFs")
plot_ic_heatmap(res_beta_etf,  "BETA Reversal — ETFs")


# In[175]:


# Helper to plot IC curves

def plot_ic_curves(results, title):
    df = pivot_mean_ic(results).sort_index()

    plt.figure()

    for col in df.columns:
        plt.plot(df.index, df[col], marker='o', label=col)

    plt.axhline(0, linestyle='--')
    plt.xlabel("Horizon")
    plt.ylabel("Mean IC")
    plt.title(title)
    plt.legend()
    plt.show()


# In[177]:


# Call

plot_ic_curves(res_raw_comm, "RAW Reversal — Commodities — IC Decay by Regime")
plot_ic_curves(res_raw_full, "RAW Reversal — Full Universe — IC Decay")


# In[179]:


# Helper to plot helper IC

def plot_rolling_ic(series, title):
    plt.figure()
    series.plot()
    plt.axhline(0, linestyle='--')
    plt.title(title)
    plt.ylabel("Rolling IC")
    plt.show()


# In[181]:


# Call

plot_rolling_ic(roll_raw_full, "Rolling IC — RAW Full")
plot_rolling_ic(roll_raw_comm, "Rolling IC — RAW Commodities")
plot_rolling_ic(roll_raw_etf,  "Rolling IC — RAW ETFs")


# In[183]:


# Helper to plot regime frequency - Basically sanity check

def plot_regime_frequency(regime_series):
    freq = regime_series.value_counts(normalize=True)

    freq.plot(kind="bar")
    plt.title("Regime Frequency")
    plt.ylabel("Fraction of Time")
    plt.show()


# In[185]:


# Call

plot_regime_frequency(regime_series)


# In[187]:


# Helper to plot regimes over time

def plot_regime_timeline(regime_series):
    mapping = {r:i for i, r in enumerate(regime_series.unique())}
    numeric = regime_series.map(mapping)

    plt.figure(figsize=(14,3))
    plt.scatter(regime_series.index, numeric, s=2)

    plt.yticks(list(mapping.values()), list(mapping.keys()))
    plt.title("Regime Timeline")
    plt.show()


# In[189]:


# Call

plot_regime_timeline(regime_series)


# In[191]:


# Helper to plot signal vs return

def plot_signal_vs_return(signal, returns, asset_subset=None):
    fwd = returns.shift(-1)

    x_all = []
    y_all = []

    for t in signal.index[:-1]:
        if asset_subset is None:
            x = signal.loc[t]
            y = fwd.loc[t]
        else:
            x = signal.loc[t, asset_subset]
            y = fwd.loc[t, asset_subset]

        mask = x.notna() & y.notna()

        x_all.extend(x[mask])
        y_all.extend(y[mask])

    plt.scatter(x_all, y_all, alpha=0.2)
    plt.axhline(0)
    plt.axvline(0)
    plt.title("Signal vs Future Return")
    plt.xlabel("Signal")
    plt.ylabel("Next Return")
    plt.show()


# In[193]:


# Call

plot_signal_vs_return(rev_raw_z, returns)


# In[ ]:




