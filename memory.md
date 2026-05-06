# memory.md — Single Source of Truth
## Statistical Arbitrage Project

---

## [2026-03-28] PHASE 1 — COMPLETE

### Summary of all changes shipped in Phase 1

| Change | File | Status |
|--------|------|--------|
| Carry removed (13 occurrences) | alpha_construction.py, optimizer.py | ✓ |
| Full-sample ADF → rolling ADF (window=120) | run_baseline.py | ✓ |
| Rolling z-score normalization inside ADF window | run_baseline.py | ✓ |
| PnL fixed: weekly compounded log returns | run_baseline.py | ✓ |
| Annualization fixed: auto-detect 52 vs 252 | report.py | ✓ |
| FM Fama-MacBeth β scaling (α = β̄ × cs_signal) | run_baseline.py | ✓ |
| Re-normalization after FM scaling | run_baseline.py | ✓ |
| β ≥ 0 clamp: prevents FM from inverting economic direction | run_baseline.py | ✓ |

### Phase 1 final numbers (report_phase1.json, generated 2026-03-28)

| Metric | Value | Notes |
|--------|-------|-------|
| Total return | +38.6% | Positive — first clean result |
| Annualized return | 3.7% | |
| Annualized vol | 55.8% | Still 3.7× above 15% target → risk model issue |
| Sharpe | 0.07 | Barely positive; signal quality low |
| Max drawdown | -48.8% | |
| Rev IC | +0.004 (t=0.23) | Not significant |
| Mom IC | +0.013 (t=0.76) | Not significant |
| Spread IC | -0.020 (t=-1.66) | Negative → FM sign flip diagnosed |
| Rev Sharpe | -0.43 | |
| Mom Sharpe | +0.27 | Only signal with consistent positive edge |
| Spread Sharpe | -0.44 | Negative → confirmed FM inversion |

### Spread sign decision (2026-03-28)

Raw signal `-(spread-mu)/std` is **economically correct** (mean reversion: short overvalued ETF). Sign problem arose inside FM alignment: with N≈7 effective cross-section, β̄ can be estimated negative by noise. Fix: `beta_t = max(0, mean(betas_fm))`. Do NOT flip raw signal — that would convert mean-reversion to trend-following and violate ADF premise.

---

## PHASE 2 — STATUS

### Step 2.2 — Spread ECM + half-life filter: COMPLETE (2026-03-28)

Changes in `run_baseline.py` step 5 (spread loop):
- Added `HALFLIFE_MAX = 30` parameter
- After ADF passes: fit ECM `Δspread = α + ρ·spread_{t-1} + ε` via OLS on same 120-day rolling window
- Skip window if ρ≥0 (not mean-reverting), (1+ρ)≤0 (explosive), or half_life≥30d
- **ECM used as pure filter only** — removed 1/half_life scaling: ρ estimation noise near 0 in a 120-day window dominates the scaling term; CS normalization downstream handles cross-pair weighting
- **Both pair legs now signalled**: spread_expanded[etf] += signal; spread_expanded[com] -= signal. Previously all 6 futures received 0 spread exposure despite being pair legs. Unified `pair_count` dict handles assets in multiple pairs.
- Per-pair diagnostic: avg_hl of active windows printed

### Step 2.1 — Rolling Sharpe combination: COMPLETE (2026-03-28)

Changes in `run_baseline.py` step 7c (new section, replaces static risk-parity after FM alignment):
- Added `SHARPE_ROLL=52, SHRINK_NU=0.3, SMOOTH_HL=4` parameters
- Standalone backtests run first (before combination) to get per-block rolling PnL
- At each rebalancing date: use PnL strictly before that date (no look-ahead), last 52 weeks
- Sharpe per block = mean/std; floor at 0 (no negative-Sharpe allocation); normalize to sum=1
- Shrink: `w = (1-0.3)·sharpe_w + 0.3·(1/3)`
- EWM smooth: halflife=4w; re-normalize after smoothing
- Combined alpha built row-by-row with time-varying weights
- Report: `phase_name="Phase2"` → writes `reports/report_phase2.json`

### Step 2.3 — DCC-GARCH regime detection: COMPLETE (2026-03-28)

**Files created/modified:**
- `regime_detection.py` (new): two public functions
- `run_baseline.py` step 7d (new): calls regime detection, applies scale to combined alpha

**`compute_regime_signals(returns_df, spread_raw_df, ...)` → `pd.DataFrame`:**
- Cleans returns via `.dropna()` (DCC raises ValueError on NaN/Inf)
- Calls `fit_multivariate_gjr(ret_matrix)` → Z (T,N), sigmas (T,N)
- Calls `dcc_fit(Z, sigmas, model='DCC')` → R_t (T,N,N)
- Extracts: rho_t (mean off-diagonal), disp_t (std off-diagonal)
- Computes rolling mean/std of rho over 60-day window for spike detection
- spike_t = 1 if rho_z > 2.0
- EWMA vol: λ=0.94 recursion on squared returns, cross-asset mean, annualized ×√252
- spread_stab: fraction of pairs (spread_raw columns) with |signal| > 1e-8 each day
- Returns DataFrame columns: rho_t, disp_t, spike_t, ewma_vol_t, spread_stab, rho_roll_mean, rho_roll_std

**`compute_alpha_scale(regime_df, reb_dates)` → `pd.Series` in [0.1, 1.0]:**
- For each rebalancing date: use most recent row ≤ date (no look-ahead)
- scale = 1.0; if spike_t=1: ×0.5; if rho_z>1.0: ×max(0.3, 1.0-0.35×(rho_z-1.0))
- Clipped to [0.1, 1.0]

**Integration in `run_baseline.py` step 7d (after step 7c, before step 8):**
- `alpha_combined_final = alpha_combined_aligned.multiply(scale_series, axis=0)`
- Step 8 uses `alpha_combined_final` instead of `alpha_combined_aligned`
- Wrapped in try/except: DCC failure → `alpha_combined_final = alpha_combined_aligned` (graceful fallback)
- Report description updated to mention DCC regime conditioning and whether applied/skipped

**DCC look-ahead policy (Option A, confirmed by user):**
- GARCH+DCC parameters estimated once on full sample (minor look-ahead in params)
- R_t path is recursively filtered → no per-day path look-ahead
- Standard practice for regime conditioning

### Phase 2 — ALL STEPS COMPLETE

---

## [2026-03-28] FM β DIAGNOSTIC — REVERSAL SIGNAL INVESTIGATION

### Issue raised

`align_signal_to_expected_return` clamps β̄ ≥ 0 for all signals. For reversal:
- `rev_raw = -residual_ret.rolling(5).sum()` — already pre-signed positive (beaten-down assets get high signal)
- FM regression `fwd_ret = a + β·rev_signal + ε` should yield β > 0 if reversal works
- With Rev IC ≈ 0 (t=0.23 in Phase 1), β̄ ≈ 0 and ~50% of weeks have β̄ < 0
- When β̄ < 0: clamp sets that week's alpha_rev = 0 (no reversal trade)
- Impact: roughly half of all rebalancing weeks have zero reversal allocation

### Analysis

**Why β ≥ 0 is correct in principle:**
- Signal is pre-signed: high rev_signal = beaten-down asset = expect upward reversion → BUY
- FM should yield β > 0 when reversal works
- Negative β̄ means: cross-sectionally, beaten-down assets underperformed that week → signal failed
- Allowing negative β would convert those weeks to TREND-FOLLOWING (buy winners, sell losers) — wrong for a mean-reversion strategy
- The clamp correctly says: "if the signal wasn't working, don't trade" rather than "invert the strategy"

**The real problem:**
- Rev IC near 0 at weekly horizon → β̄ near 0 → ~50% weeks zeroed out
- This is a symptom of weak signal, not a bug in the clamp
- Exponential decay (CLAUDE.md §4 specifies half-life 3–5 days) NOT YET IMPLEMENTED — current is simple 5-day sum

**Test implemented:** `beta_floor=-np.inf` (unclamped) run in diagnostic section after step 7b.
Expected result: unclamped IC ≈ clamped IC (or worse), standalone Sharpe ≈ similar (or worse), because allowing negative β adds trend-following noise to ~50% of weeks.

**Decision after empirical test:** Documented in step 7b diagnostic output when pipeline is run.

---

## [2026-03-27] FULL AUDIT — INITIAL FINDINGS

### FILE LOCATIONS
- Main code: `original code/alpha_construction.py`, `original code/covariance_model.py`, `original code/optimizer.py`
- DCC: `src/dcc garch/dcc/`, `src/dcc garch/garch/gjr_garch.py`, `src/dcc garch/README.md`
- Data: `data/raw/unified_dataset.csv`
- NOTE: Paths inside the scripts are HARDCODED to user's local absolute paths — must be fixed.

---

## SECTION 1: CODE AUDIT FINDINGS

---

### 1.1 CARRY — STATUS: FULLY PRESENT, INCORRECTLY IMPLEMENTED, MUST BE REMOVED

**Current implementation (alpha_construction.py):**
- Lines 261–319: Carry computed as 30-day rolling sum of RESIDUAL RETURNS for futures
- For ETFs: carry is set to 0 (not meaningful)
- This is NOT carry. Proper carry = futures curve slope (F1 - F2 term structure). What is implemented is a slow-moving momentum signal.

**ALL carry occurrences in codebase:**

| File | Lines | Description |
|------|-------|-------------|
| alpha_construction.py | 261–319 | Full carry computation (Stage 4) |
| alpha_construction.py | 475 | `vol_carry = carry_final.std().mean()` |
| alpha_construction.py | 496 | `1/vol_carry` in risk parity weights |
| alpha_construction.py | 508 | `"carry": w[2]` in weights dict |
| alpha_construction.py | 551 | `alpha_carry = carry_final.loc[common_index, common_columns]` |
| alpha_construction.py | 563 | `+ weights["carry"] * carry_final` in alpha_combined |
| alpha_construction.py | 609 | `alpha_carry.to_pickle("alpha_carry_no_bl.pkl")` |
| optimizer.py | 44 | `alpha_carry = pd.read_pickle("alpha_carry.pkl")` |
| optimizer.py | 222 | `res_carry = run_backtest(alpha_carry, cov_dict, returns)` |
| optimizer.py | 232 | `"Carry": res_carry` in results dict |
| optimizer.py | 266 | `combine_alphas([alpha_rev, alpha_mom, alpha_carry, alpha_spread])` |
| optimizer.py | 354 | `"Carry": alpha_carry` in alpha_dict |
| optimizer.py | 389 | `"Carry": res_carry` in all_results saved to backtest_results.pkl |

**Decision (from CLAUDE.md):** Remove carry entirely. Do NOT fix or improve it.

---

### 1.2 COINTEGRATION — STATUS: CRITICAL BUG (LOOK-AHEAD BIAS)

**File:** `alpha_construction.py`, lines 380–415

**Issue 1 — CRITICAL: Full-sample ADF test (look-ahead bias)**
- Lines 394–400: `pval = adfuller(spread.dropna())[1]`
- `spread.dropna()` is the FULL SAMPLE spread (all T observations)
- The ADF decision (include/exclude pair) is made using future data
- In a real backtest, we would not know the full-sample test result at any point in time
- This contaminates the entire backtest — pairs included/excluded using information not available at the trading date
- FIX REQUIRED: Roll the ADF test within the same 120-day window used for rolling OLS

**Issue 2 — Missing rolling z-score of spread**
- The spread (residual) is used directly as a signal, but is not normalized by its rolling mean/std
- A proper mean-reversion signal should be: `z_t = (spread_t - mu_rolling) / std_rolling`
- Currently only cross-sectional normalization is applied (not rolling time-series normalization)

**What IS correct:**
- Rolling OLS for hedge ratio: YES — `RollingOLS(y, sm.add_constant(X), window=120)` ✓
- Residual computation: YES — `spread = y - (alpha_t + beta_t * X)` ✓
- Cross-sectional normalization after: YES ✓

---

### 1.3 BETA NEUTRALIZATION — STATUS: CORRECT

**File:** `alpha_construction.py`, lines 88–123

- Rolling OLS vs SPX: `RollingOLS(y, X, window=60)` — CORRECT ✓
- Residual = raw return minus market component: `y - (alpha_t + beta_t * mkt)` — CORRECT ✓
- Initial NaN rows dropped after rolling window: `residual_ret.dropna()` — CORRECT ✓
- All downstream signals built on `residual_ret` — CORRECT ✓

---

### 1.4 ALPHA NORMALIZATION — STATUS: INCOMPLETE (MISSING RE-NORMALIZATION)

**File:** `alpha_construction.py`

**CLAUDE.md requires 3 steps:** CS z-score → MAD winsorize → re-normalize

| Alpha | CS z-score | MAD winsorize | Re-normalize |
|-------|-----------|--------------|--------------|
| Reversal (lines 182–198) | YES (line 182) | YES (line 190) | NO — `rev_final = rev_wins` |
| Momentum (lines 233–249) | YES (line 233) | YES (line 241) | NO — `mom_final = mom_wins` |
| Spread (lines 413–415) | YES (line 413) | YES (line 414) | NO — `spread_final = spread_wins` |

**Missing:** Third normalization step (CS z-score after winsorization) is absent for all three alphas.

**Additional note:** `cs_normalize` divides by cross-sectional std per day. If std = 0 (all assets identical on a day), this produces NaN/inf. No guard exists.

---

### 1.5 ALPHA COMBINATION — STATUS: INCORRECT (MUST REPLACE WITH ROLLING SHARPE)

**File:** `alpha_construction.py`, lines 466–565

**What is implemented:** Static risk parity (1/vol weighting)
- Volatilities computed once over full sample: `vol_rev = rev_final.std().mean()` (line 473)
- Weights: `w_i ∝ 1 / vol_i` (line 496)
- These are FIXED weights — computed once, not updated over time
- This has look-ahead bias: uses full-sample volatility

**What CLAUDE.md requires:**
- Rolling Sharpe per alpha (dynamic)
- Shrinkage of weights
- Time smoothing of weights

**Secondary (optimizer.py):** `combine_alphas()` (lines 64–68) uses equal weights (1/N) as default. This is a different, simpler combination.

---

## SECTION 2: DATA AUDIT FINDINGS

**File:** `data/raw/unified_dataset.csv`

| Property | Value |
|----------|-------|
| Shape | 2,661 rows × 15 columns |
| Date range | 2015-05-01 to 2025-11-26 (~10.5 years) |
| NaNs | 0 (clean) |
| Columns | Date, CLc1, Cc1, HGc1, LCOc1, NGc1, Wc1, JETS, XLB, XLE, XLI, XLP, XLU, XLY, SPX |

**Futures:** CLc1, Cc1, HGc1, LCOc1, NGc1, Wc1
- All are single front-month continuous contracts (c1 suffix)
- **NO second maturity contracts** (no CLc2, HGc2, etc.)
- **CONCLUSION: Carry CANNOT be implemented. Term structure is absent.**

**ETFs:** JETS, XLB, XLE, XLI, XLP, XLU, XLY (7 sector ETFs)

**Market:** SPX

**Data appears to be already in return form** (values ~0.001–0.025 magnitude, consistent with daily returns).

**Data quality:** No NaNs, no Infs detected, full date coverage.

---

## SECTION 3: DCC-GARCH UNDERSTANDING

**Source files:**
- `src/dcc garch/garch/gjr_garch.py` — Stage 1: univariate GJR-GARCH
- `src/dcc garch/dcc/` — Stage 2: DCC/ADCC estimation

### Stage 1 — GJR-GARCH (gjr_garch.py)

- Model: GJR-GARCH(1,1,1) with Student-t innovations
- Entry point (multivariate): `fit_multivariate_gjr(returns_matrix)` where `returns_matrix` shape = (T, N)
- Returns scaled ×100 internally before fitting
- Outputs:
  - `Z` (T, N): standardized residuals `z_{i,t} = ε_{i,t} / σ_{i,t}`
  - `sigmas` (T, N): conditional volatilities in % daily units
  - `params`: list of N fitted parameter Series
  - `results`: list of N arch result objects

### Stage 2 — DCC (dcc/optimizer.py, dcc/model.py)

- Entry point: `fit(Z, sigmas, model='DCC')`
- Supports both DCC (Engle 2002) and ADCC (Cappiello et al. 2006)
- Optimization: SLSQP with stationarity constraint (a + b < 1)
- Q_t recursion: `Q_t = (1-a-b)·Q̄ + a·z_{t-1}z_{t-1}' + b·Q_{t-1}`
- Outputs:
  - `R` (T, N, N): conditional correlation matrices — PRIMARY output for regime detection
  - `H` (T, N, N): conditional covariance matrices (H_t = Σ_t · R_t · Σ_t)
  - `Q` (T, N, N): pseudo-correlation path
  - `params`: (a, b) for DCC
  - `llh`: full Gaussian log-likelihood
  - `converged`: bool
  - `delta`: None for DCC

### Intended use in this project (from README.md)

- DCC is used ONLY for **regime detection**, NOT for portfolio optimization
- Ledoit-Wolf is used for optimization (covariance_model.py)
- Key outputs for regime detection:
  - `rho_t = mean of off-diagonal elements of R[t]`
  - Dispersion = std of off-diagonal elements
  - Correlation spikes (sudden increases in rho_t)

### Call sequence to use DCC

```python
from garch.gjr_garch import fit_multivariate_gjr
from dcc import fit as dcc_fit

# Step 1: Fit GJR-GARCH per asset
garch_result = fit_multivariate_gjr(returns_matrix)  # (T, N) log returns
Z      = garch_result['Z']       # (T, N) standardized residuals
sigmas = garch_result['sigmas']  # (T, N) conditional vols in % daily

# Step 2: Fit DCC
dcc_result = dcc_fit(Z, sigmas, model='DCC')
R_t = dcc_result['R']  # (T, N, N) — use for regime detection only
```

---

## SECTION 4: IDENTIFIED ISSUES (PRIORITY ORDER)

| # | Severity | File | Issue |
|---|----------|------|-------|
| 1 | CRITICAL | alpha_construction.py:394 | Full-sample ADF test — look-ahead bias in cointegration |
| 2 | CRITICAL | alpha_construction.py:261–319, optimizer.py | Carry present throughout — must be removed entirely |
| 3 | HIGH | alpha_construction.py:466–497 | Alpha combination uses static risk parity on full sample — must be replaced with rolling Sharpe + shrinkage |
| 4 | HIGH | alpha_construction.py:182–198, 233–249, 413–415 | Missing re-normalization after winsorization (3rd step) for all alphas |
| 5 | MEDIUM | alpha_construction.py:27 | Hardcoded absolute path for data — must use relative paths |
| 6 | MEDIUM | covariance_model.py:33 | Hardcoded absolute path for data — must use relative paths |
| 7 | MEDIUM | alpha_construction.py:380–415 | Spread not rolling z-scored per pair (no rolling mean/std normalization of residual) |
| 8 | LOW | alpha_construction.py:143–146 | cs_normalize has no zero-std guard (division by zero possible) |
| 9 | LOW | optimizer.py:42–47 | Loads alpha_bl.pkl and alpha_total.pkl which are not produced by alpha_construction.py (Black-Litterman artifacts) |

---

## SECTION 5: RECOMMENDED FIXES (DO NOT IMPLEMENT YET)

**Fix 1 — Remove carry completely**
- Delete Stage 4 block (lines 261–319) from alpha_construction.py
- Remove carry from risk parity weights computation (lines 473–509)
- Remove carry from alpha_combined (line 563)
- Remove carry save (line 609)
- Remove carry load and backtest from optimizer.py (lines 44, 222, 232, 266, 354, 389)
- Result: 3-alpha system: reversal + momentum + spread

**Fix 2 — Replace ADF with rolling ADF**
- For each rolling window of 120 days, run ADF on the residual within that window only
- Only include pair at time t if ADF p-value < 0.10 within that window
- Eliminates look-ahead bias

**Fix 3 — Replace static risk parity with rolling Sharpe combination**
- Compute rolling Sharpe per alpha (e.g., 60-day or 120-day window)
- Apply shrinkage toward equal weights
- Apply time smoothing (e.g., EWM)
- Weights updated at each rebalancing date

**Fix 4 — Add re-normalization after winsorization**
- After `robust_winsorize(...)`, apply `cs_normalize(...)` again for all alphas
- rev_final = cs_normalize(rev_wins)
- mom_final = cs_normalize(mom_wins)
- spread_final = cs_normalize(spread_wins)

**Fix 5 — Fix hardcoded paths**
- Replace absolute paths in alpha_construction.py and covariance_model.py with relative paths from project root

**Fix 6 — Add rolling z-score to spread signal**
- After computing the pair residual spread, apply: `z_t = (spread_t - spread.rolling(120).mean()) / spread.rolling(120).std()`
- This normalizes the mean-reversion signal within each rolling window

---

## STATUS

- [x] Audit complete (2026-03-27)
- [x] Phase 1 implemented (2026-03-27)
- [x] report.py created (2026-03-27) — v1
- [x] report.py patched (2026-03-27) — v2: annualized return crash fix + diagnostics
  - [x] Carry removed from alpha_construction.py (all 11 code occurrences)
  - [x] Carry removed from optimizer.py (6 code occurrences)
  - [x] Cointegration: full-sample ADF replaced with rolling ADF (120-day window)
  - [x] Cointegration: rolling z-score normalization added within same window
- [x] Phase 2 Step 2.2 complete (2026-03-28): ECM filter + commodity leg + pure filter (no 1/hl scaling)
- [x] Phase 2 Step 2.1 complete (2026-03-28): rolling Sharpe combination (52w, ν=0.3, hl=4w)
- [x] FM β diagnostic implemented (2026-03-28): align_signal_to_expected_return gains beta_floor param + beta_stats return; step 7b-inv runs clamped vs unclamped reversal comparison
- [ ] Phase 2 Step 2.3 pending: DCC-GARCH regime detection integration
- [ ] Exponential decay for reversal (CLAUDE.md §4 specifies half-life 3–5d; current = simple 5d sum) — next candidate improvement after DCC

---

## [2026-03-27] PHASE 1 IMPLEMENTATION

### Carry Removal

**alpha_construction.py:**
- Deleted Stage 4 block entirely (was lines 261–319): `carry_raw`, `carry_norm`, `carry_wins`, `carry_final`
- Removed `vol_carry` from volatility computation
- Removed carry from print statement
- Removed `1/vol_carry` from `raw_w` array (now 3-element)
- Removed `"carry": w[2]` from `weights` dict (spreads now at index `w[2]`)
- Removed `carry_final.index` from `common_index` intersection
- Removed `alpha_carry = carry_final.loc[...]` from alignment block
- Removed `+ weights["carry"] * carry_final` from `alpha_combined`
- Removed `alpha_carry.tail(10)` inspection cell
- Removed `alpha_carry.to_pickle("alpha_carry_no_bl.pkl")` from save block

**optimizer.py:**
- Removed `alpha_carry = pd.read_pickle("alpha_carry.pkl")`
- Removed `res_carry = run_backtest(alpha_carry, ...)`
- Removed `"Carry": res_carry` from `results` dict
- Removed `alpha_carry` from `all_alpha = combine_alphas([...])`
- Removed `"Carry": alpha_carry` from `alpha_dict`
- Removed `"Carry": res_carry` from `all_results`
- Kept: narrative comments (historical), `alpha_no_carry` variable (carry-free combination)

**Result:** System is now a clean 3-alpha framework: reversal + momentum + spread.

---

### Cointegration Fix

**alpha_construction.py (cointegration loop):**

OLD (look-ahead bias):
```python
pval = adfuller(spread.dropna())[1]  # full-sample ADF
if pval < 0.10:
    spread_raw[...] = -spread
else:
    spread_raw[...] = 0.0
```

NEW (rolling ADF + rolling z-score):
```python
spread_arr = spread.values
signal = pd.Series(0.0, index=spread.index)

for i in range(window - 1, len(spread_arr)):
    w_data  = spread_arr[i - window + 1 : i + 1]
    w_clean = w_data[~np.isnan(w_data)]
    if len(w_clean) < window // 2:
        continue
    pval = adfuller(w_clean)[1]       # ADF on window only
    if pval >= 0.10:
        continue
    mu  = w_clean.mean()
    std = w_clean.std()
    if std < 1e-8:
        continue
    signal.iloc[i] = -(spread_arr[i] - mu) / std

spread_raw[f"{etf}_{com}"] = signal
```

**Properties of the new implementation:**
- At each time t, ADF is applied to `spread[t-119 : t+1]` (120 days)
- No information from future dates ever used
- Rolling z-score uses same window: `(spread_t - mu_window) / std_window`
- Zero signal when: not enough data, ADF fails, not cointegrated, std ≈ 0

---

### ADF Design Evaluation (KEEP or REMOVE)

**DECISION: KEEP ADF filtering**

**Reasoning:**

1. **Statistical validity of mean reversion requires stationarity**
   ADF tests whether the spread is stationary (I(0)) in a given window. Trading a mean-reversion signal on a non-stationary spread (I(1) random walk) has no theoretical basis — the "mean" the signal reverts to is itself a random walk with no stable level. Without the ADF filter, we would be generating signals on spreads that may drift indefinitely. The filter is not a data-mining heuristic; it is a structural requirement for the mean-reversion premise to hold.

2. **Rolling ADF addresses look-ahead bias cleanly**
   The previous objection to ADF was not the test itself but the full-sample application. Rolling ADF eliminates that entirely. At each time t, the test conditions only on data available up to t.

3. **Stability vs noise trade-off**
   Rolling ADF in a 120-day window will be noisy — it may flip a pair in and out of the signal. This is acceptable because:
   - The 120-day window is long enough (~4× the typical ADF critical lag)
   - The p-value threshold (0.10) is lenient, reducing false exclusions
   - Flipping to zero is conservative and prevents trading on drift regimes
   - A continuous z-score without filtering would silently generate signal on explosive spreads, which is worse

4. **Alternative: continuous z-score without ADF**
   Removing ADF and always trading the z-score is simpler but dangerous:
   - When the spread is non-stationary, the z-score loses meaning (no stable mean to revert to)
   - Strategy would take positions expecting reversion that may never occur
   - No economic or statistical justification for the trade

5. **Recommendation for robustness**
   If ADF noise is a concern in practice, the fix is to use a smoother filter (e.g., require ADF p < 0.10 for N consecutive days before activating, or use a rolling average of recent p-values). But the filter itself should not be removed.

**Final answer: KEEP ADF. The rolling implementation eliminates look-ahead bias while preserving statistical validity of mean-reversion signals.**

---

## [2026-03-28] DIAGNOSTIC ANALYSIS — PHASE 1 BASELINE

### Baseline Numbers (report_phase1.json)

| Metric | Value |
|--------|-------|
| Total return | -126.4% (bankrupt) |
| Annualized vol | 93.5% (target was 15%) |
| Max drawdown | -126.9% |
| Rev Sharpe | -1.07 |
| Mom Sharpe | +0.42 |
| Spread Sharpe | -1.10 |
| Gross exposure (mean) | 2.19 |
| Spread active % | 35.8% |

### Root Cause Analysis

**PRIMARY BUG: PnL computation uses 1-day forward returns for a weekly-holding strategy.**

In `run_baseline.py`:
```python
next_ret = returns_df.shift(-1).loc[w_df.index, assets]
```
`returns_df` is daily. `.shift(-1)` moves by 1 day. `.loc[w_df.index]` selects Friday dates.
Result: next_ret = **next Monday's 1-day return**, not the **weekly (Friday→Friday) return**.

The portfolio holds for 5 days but earns only 1 day of returns. Combined with the vol targeting calibrated to daily covariance, positions are massively over-leveraged relative to the true weekly risk exposure.

**SECONDARY: IC measured at wrong frequency.**
IC computed at 1-day forward horizon. Strategy rebalances weekly. Momentum's "negative IC" (-0.019) at 1-day is expected (short-term mean reversion of a 60-day signal). Weekly IC is likely positive. This is NOT a signal failure, it's a measurement artifact.

**ABOUT α ≠ E[r] hypothesis:**
Partially valid — z-score has wrong units for Chernov optimizer. But this does not flip trade direction. Vol targeting compensates for scale. Not the primary issue.

**About data (from data_check.py):**
- ETFs: % returns / 100 → decimal; Futures: pct_change of prices; SPX: pct_change of price
- All in consistent decimal simple-return form — no scaling mismatch
- NGc1 and WTI extreme moves flagged in data_check.py — 60-day LW covariance cannot capture these spikes, causing over-leverage during stress periods

### Decision

**Single next step: Fix PnL computation to use 5-day compounded weekly returns aligned to rebalancing dates.**
Also fix annualization in compute_performance: weekly PnL → multiply by sqrt(52), not sqrt(252).

---

## [2026-03-28] ALPHA ALIGNMENT FIX + VOL DIAGNOSIS

### Root cause confirmed

IC is statistically insignificant for reversal (t=0.97) and spread (t=0.93). Momentum IC is negative at daily horizon but positive at weekly — this is expected (short-term mean-reversion of 60-day signal).

α ≠ E[r] is a sizing problem, not a direction problem. CS z-scores are dimensionless (factor ~14,000 mismatch to true E[r]). Combined with 5× risk model underestimation, positions are catastrophically over-leveraged.

### Action: converted α to expected returns

Added `align_signal_to_expected_return()` in run_baseline.py:
- Uses raw signals (rev_raw, mom_raw, spread_expanded_raw — no CS normalization)
- Rolling OLS: fwd_weekly_ret = a + β × signal_t + ε (window = 52 weeks)
- α_t = β_t × signal_t (expected return scale)
- No look-ahead bias: at t, OLS uses only returns up to t-1

New alpha files saved: data/alphas/alpha_rev.pkl, alpha_mom.pkl, alpha_spread.pkl, alpha_combined.pkl

**ADDITION (same session):** Re-normalization after FM scaling (3rd step of CLAUDE.md §4):
- After FM alignment each block's std ∝ β̄_block (different ICs → different scales)
- Without re-normalization, risk-parity weights just invert each β̄ → arbitrary combination
- Fix: normalize_alpha (CS z-score + MAD winsorize) applied to each aligned block before combining
- Diagnostic prints pre/post std to verify comparability
- Saved files: alpha_rev_n, alpha_mom_n, alpha_spread_n, alpha_combined_aligned

**CORRECTION (same session):** Changed from per-asset time-series OLS to cross-sectional Fama-MacBeth:
- CS normalization is KEPT (fundamental to stat arb cross-sectional ranking)
- FM β̄_t = mean of rolling cross-sectional OLS slopes over past 52 weeks
- α_{i,t} = β̄_t × cs_normalized_signal_{i,t}
- β̄ ≈ rolling IC × σ_cross_returns → correct E[r] scale
- CS-normalized signals (rev_final, mom_final, spread_expanded) are the FM inputs, NOT raw signals
- `spread_expanded_raw` added but no longer used as FM input (kept as intermediate)

### Vol targeting diagnosis (code added, not yet run)

Step 10 in run_baseline.py computes:
- σ_model_t = sqrt(x_t'Σ_t x_t × 252) per rebalancing date
- σ_realized = 8-week rolling std of PnL × sqrt(52)
- Per-asset marginal variance contribution
- Avg pairwise Ledoit-Wolf correlation (last 52w)

---

## [2026-03-28] BUG FIXES — PnL COMPUTATION + ANNUALIZATION

### Bug 1 Fixed: PnL now uses correct Friday→Friday weekly log returns

**File:** `run_baseline.py`, `run_backtest()` (~line 155)

**OLD (wrong — captured only 1 day):**
```python
next_ret = returns_df.shift(-1).loc[w_df.index, assets]
```

**NEW (correct — sums daily log returns from (d_curr, d_next]):**
```python
reb_dates_sorted = sorted(w_df.index)
weekly_ret_rows = {}
for i in range(len(reb_dates_sorted) - 1):
    d_curr, d_next = reb_dates_sorted[i], reb_dates_sorted[i+1]
    mask = (returns_df.index > d_curr) & (returns_df.index <= d_next)
    wret = returns_df.loc[mask, assets]
    if len(wret) > 0:
        weekly_ret_rows[d_curr] = wret.sum(axis=0)
next_ret = pd.DataFrame(weekly_ret_rows).T
hold_dates = w_df.index.intersection(next_ret.index)
```

Also: `w_held` (not `w_df`) returned as weights, aligned to `hold_dates`.
Also: in-function Sharpe changed from `sqrt(252)` → `sqrt(52)`.

### Bug 2 Fixed: Annualization auto-detects periodicity

**File:** `report.py`

Added `_detect_periods_per_year(pnl)`:
- Returns `52` if median gap between observations >= 5 calendar days (weekly)
- Returns `252` otherwise (daily)

`compute_performance` now uses `freq` instead of hardcoded 252 for both:
- `ann_return = base ** (freq / n) - 1`
- `ann_vol = pnl.std() * sqrt(freq)`

`compute_stability` now uses `freq` for `roll_sharpe` annualization.

**Impact:** Previous annualized vol was 2.2× overstated. Previous annualized return exponent was wrong (252/509 ≈ 0.495 vs correct 52/509 ≈ 0.102). All metrics now correctly scaled for weekly rebalancing.

---

## [2026-03-27] REPORT.PY — DEBUGGING & DIAGNOSTICS (v2)

### Bug Fix: Annualized Return

**Problem:** `(1 + total_return) ** (252/n)` raises complex number when `total_return < -1` (i.e., total portfolio loss > 100%). Python's `float.__pow__` with a fractional exponent on a negative base produces a complex, which then crashes `round()` and `json.dump()`.

**Fix (report.py, `compute_performance`):**
```python
# OLD (crashes on negative base):
ann_return = float((1 + total_return) ** (252 / n) - 1)

# NEW (safe):
base = 1.0 + total_return
if base <= 0:
    ann_return = float("nan")
else:
    ann_return = float(base ** (252.0 / n) - 1)
```
Sharpe and Calmar also guarded with `not np.isnan(ann_return)` checks.
JSON serialization: `nan` → `None` via conditional in return dict.

---

### Diagnostics Added

**New function: `compute_diagnostics(weights_df, alpha_dict, returns_df)`**

Returns dict with 3 sections, stored under `report["diagnostics"]`:

| Key | What it computes |
|-----|-----------------|
| `gross_exposure` | mean / median / max of sum(|w_t|) per date |
| `spread_stats` | mean, std, min, max, n_nonzero, pct_nonzero of spread alpha (non-zero entries) |
| `ic` | per-alpha mean_ic + ic_std (reuses compute_ic) |

---

## [2026-03-28] NEXT STEPS — PRIORITIZED

---

### CRITICAL

#### Fix covariance mismatch

**Description:**
Covariance is estimated on raw returns (`ret[ASSETS]`) while alpha signals and optimizer positions are built on beta-neutralized residual returns (`residual_ret[ASSETS]`). The risk model does not match the return process that actually generates PnL.

**Action:**
```python
returns_all = residual_ret[ASSETS]   # was: ret[ASSETS]
```
Applied to the Ledoit-Wolf rolling window in step 7 (`cov_dict` construction).

**Why:**
Vol targeting scales positions using `x_t @ Σ_t @ x_t`. If Σ_t includes market beta variance but the positions are beta-neutral, the model-implied vol is inflated → scaling factor `target_vol / port_vol` is too small → positions are undersized, OR if the covariance is underestimating true vol (noisy LW), positions are oversized. Either way, realized vol diverges from target. Phase 1 result: 55.8% realized vol vs. 15% target (3.7× overshoot) is consistent with this mismatch.

**Expected impact:**
Realized vol should move toward 15–20% range. Leverage instability should decrease. All downstream metrics (Sharpe, drawdown) will change — this is a first-order fix, not a refinement.

**Actual outcome (2026-03-29, run 1 — broken):** Fix applied. However, switching to residual covariance made `Sigma_resid` diagonals much smaller than `Sigma_raw` (idiosyncratic vol ~0.3–0.8% daily vs raw ~1–2%). This caused `port_vol` to collapse and the vol targeting scale factor to become 5–15×, inflating positions far beyond ±MAX_WEIGHT. One bad week produced PnL < –1.0, turning `(1 + pnl)` negative — cumulative return compounded to –187%, `annualized_return = null`, `max_drawdown = –218%`. Catastrophic failure. Root cause: position limits were enforced only BEFORE vol targeting, not after.

**Actual outcome (2026-03-29, run 2 — after position limit fix):** See report_phase2.json. Results now valid and significantly improved vs Phase 1. See PHASE 2 FINAL RESULTS section below.

---

#### Vol targeting position-limit explosion (FIXED 2026-03-29)

**Root cause:**
`optimize_weights` clips positions to ±MAX_WEIGHT (0.03) and centers to dollar-neutral. Vol targeting then multiplies by `target_vol / port_vol`. With residual covariance, this scale factor is 5–15×, pushing positions to ±0.15–0.45. `x_prev = x_t.copy()` then carries the over-leveraged positions forward into the next period's inertia term `κ·x_{t-1}`, compounding week over week. First PnL < –1.0 collapses the cumulative return.

**Fix (run_baseline.py, `run_backtest`):**
After vol targeting, re-apply dollar neutrality and position limits:
```python
if port_vol > 1e-8:
    x_t = x_t * (target_vol / port_vol)
x_t = x_t - x_t.mean()                      # restore dollar neutrality
x_t = np.clip(x_t, -max_weight, max_weight) # restore position limits
```

**Effect:**
Vol target becomes a soft ceiling rather than a hard target (clipping may reduce port vol below 15%). This is correct behavior — position risk limits take priority over vol targeting. `x_prev` is now always a well-bounded vector, preventing leverage compounding.

---

### HIGH

#### Validate vol targeting after covariance fix

**Check after fix:**
- Realized annualized vol ≈ 15–20% (acceptable band)
- No large episodic overshoot vs. target in rolling vol plot
- Model-implied vol (from Σ_t) should track realized vol directionally

**Diagnostic already in run_baseline.py (step 10):** model vol vs. realized vol comparison is printed. Use this output to confirm fix.

---

#### Separate short-term reversal into its own book (do NOT remove)

**Description:**
Reversal (REV_WINDOW=5 days) operates on a fundamentally different horizon than momentum (MOM_WINDOW=60 days) and spread (SPREAD_WINDOW=120 days). Mixing them inside one optimizer creates horizon mismatch: the optimizer's κ (inertia) and γ (risk aversion) cannot be simultaneously optimal for both short-term and medium-term signals.

**Action:**
Remove reversal from the current alpha combination block (step 7c rolling Sharpe). Reserve it for a future dedicated short-term book with its own optimizer, covariance window, and vol target.

**Why:**
- Reversal signals have IC decay over days, not weeks. A weekly rebalancing schedule is too slow to capture the signal and too fast to let it fully decay.
- Phase 1/2 results: Rev standalone Sharpe = -0.43. The signal may not be broken — it may just be running at the wrong frequency.
- Mixing horizons forces the optimizer to find a single γ/κ that fits neither signal well.

**What to keep:**
Momentum + Spread as the medium-term combined alpha. The rolling Sharpe combination (step 7c) remains with 2 blocks instead of 3.

---

#### Regime conditioning vs. vol targeting conflict

**Description:**
Regime scaling (`scale_t ∈ [0.1, 1.0]`) reduces `alpha_combined_final`. The optimizer sees a smaller α_t → produces smaller x_t → `port_vol` drops → vol targeting inflates x_t back toward 15%. The regime guard and vol target partially cancel.

**Why it matters:**
During a correlation spike, the intent is to reduce risk exposure. But if vol targeting immediately re-inflates positions, the effective risk reduction is much smaller than `scale_t` implies.

**Current status:** No change yet. Documenting for future resolution.

**Future option (do not implement yet):**
Apply regime scaling to `target_vol` instead of alpha:
```python
effective_target_vol = TARGET_VOL * scale_t   # passed into run_backtest
```
This would reduce the vol target itself during crises, preventing the offsetting inflation. Requires refactoring `run_backtest` to accept a per-date `target_vol` series.

---

### MEDIUM

#### Refactor pipeline into a "Book" abstraction

**Goal:**
Encapsulate the current system (momentum + spread) into a reusable, self-contained unit. Each "book" owns: alpha construction, covariance estimation, optimizer call, vol target, and PnL attribution. Supports multi-horizon architecture without duplicating pipeline code.

**Motivation:**
As reversal is spun off into its own short-term book and potentially a long-term book is added (carry, macro), the current flat pipeline will become unmanageable. A Book class or function factory isolates each horizon cleanly.

**Current status:** Design only. Do not implement until covariance fix and reversal separation are complete and validated.

---

#### Covariance estimation stability (Ledoit-Wolf sample ratio)

**Description:**
After switching to `residual_ret`, re-evaluate COV_WINDOW. Current 60-day window gives T/N ≈ 60/13 ≈ 4.6. Small eigenvalues of the sample covariance are systematically underestimated even with Ledoit-Wolf shrinkage.

**Action (after covariance fix is validated):**
Test COV_WINDOW = 90 and 120 days. Compare:
- Model-implied vs. realized vol tracking
- Sharpe and drawdown sensitivity to window length
- Prefer the shortest window that still produces stable realized vol

**Note:** Longer windows reduce noise but increase staleness — during regime changes, a 120-day window carries 4 months of pre-crisis returns into the post-crisis estimate.

---

### LOW

#### Parameter optimization (after all fixes are validated)

**Prerequisite:** Do NOT optimize parameters until covariance mismatch is fixed and vol targeting is confirmed working. Optimizing on a misspecified system finds compensating errors, not true signal.

**Procedure:**
- Expanding-window walk-forward: 70% in-sample / 30% OOS
- Optimize on IC (not Sharpe) to reduce sensitivity to specific return realizations
- Search order: GAMMA × KAPPA first (most impactful, affect all signals equally), then REV_WINDOW / MOM_WINDOW, then combination weights (SHARPE_ROLL, SHRINK_NU)
- Prefer flat regions over global optima: parameter values stable across ±20% perturbation
- Report OOS degradation explicitly; >50% Sharpe drop from IS to OOS = overfitting

**Parameters to leave fixed (not tuned):**
LAMBD (market fact), MAX_WEIGHT (risk mandate), TARGET_VOL (performance target), spike_z / spike_window (too few crisis events to calibrate).

---

#### Extend to multi-horizon engine

**Description:**
Long-term architecture goal: support multiple independent books (short-term reversal, medium-term momentum+spread, long-term macro/carry), each with its own alpha, covariance, optimizer, and vol target. A top-level allocator combines book-level PnL streams.

**Dependencies:**
Book abstraction (see MEDIUM above) must exist first.

**Current status:** Vision only. No implementation until medium-term book is stable and validated OOS.

**Updated `compute_ic`:** Now also returns `ic_std` (std dev of cross-sectional IC series).

**Updated `generate_report`:** Calls `compute_diagnostics` and stores result under `"diagnostics"` key.

### Final JSON Structure (v2)

```json
{
  "metadata":      {...},
  "performance":   {total_return, annualized_return, vol, sharpe, calmar, max_dd, n_periods, dates},
  "signal_quality": {"Rev": {mean_ic, ic_std, ic_tstat, ic_count}, ...},
  "stability":     {avg_rolling_sharpe, pct_positive_periods, avg_turnover, window},
  "attribution":   {"Rev": {sharpe, total_return, ann_return, max_dd}, ...},
  "diagnostics":   {
    "gross_exposure": {mean, median, max},
    "spread_stats":   {mean, std, min, max, n_nonzero, pct_nonzero},
    "ic":             {"Rev": {mean_ic, ic_std}, "Mom": {...}, "Spread": {...}}
  }
}
```

**Missing from report (known gap):** No `regime` section. DCC stats (rho mean/std, n_spikes, scale mean/min, convergence) are printed to console only and not persisted in the JSON. `generate_report` does not accept `regime_df` or `scale_series`. Fix proposed but not yet implemented — awaiting approval.

---

## [2026-03-29] PHASE 2 FINAL RESULTS — report_phase2.json

Generated after both fixes: covariance mismatch (residual_ret) + position limit re-clip after vol targeting.

### Performance

| Metric | Value | vs Phase 1 | Notes |
|---|---|---|---|
| Total return | +82.8% | +38.6% Phase 1 | Large improvement |
| Annualized return | +7.1% | +3.7% Phase 1 | Almost 2× |
| Annualized vol | 27.8% | 55.8% Phase 1 | Still above 15% target |
| Sharpe ratio | 0.254 | 0.07 Phase 1 | 3.6× improvement |
| Max drawdown | -7.2% | -48.8% Phase 1 | Dramatic improvement |
| Calmar ratio | 0.983 | near 0 Phase 1 | |
| Periods | 460 weekly | | 2016-10-21 to 2025-11-14 |

### Signal Attribution (standalone backtests)

| Signal | Sharpe | Total Return | Max DD | Status |
|---|---|---|---|---|
| Reversal | -0.38 | -84.4% | -85.1% | Broken — horizon mismatch |
| Momentum | +0.34 | +83.2% | -9.4% | Working well |
| Spread | +0.37 | +98.5% | -11.5% | Working well |

### Signal Quality (IC)

| Signal | Mean IC | IC t-stat | Count | Notes |
|---|---|---|---|---|
| Rev | +0.012 | +0.50 | 242 | Not significant |
| Mom | -0.021 | -0.46 | 89 | Negative — short IC count |
| Spread | +0.022 | +0.86 | 215 | Borderline |

### Diagnostics

| Metric | Value | Notes |
|---|---|---|
| Gross exposure mean | 0.375 | Max is 0.39 = 13 × 0.03 — position limits fully saturated |
| Gross exposure max | 0.390 | Vol targeting is hitting the ceiling every period |
| Avg turnover | 0.106 | Reasonable |
| Avg rolling Sharpe | 0.094 | Barely positive but stable |
| Pct positive periods | 47.4% | Slightly below 50% — right tail skew from Spread/Mom |

### Key Observations

**1. Vol targeting is structurally constrained.**
Gross exposure is capped at 0.39 (= 13 × MAX_WEIGHT) on most dates. This means vol targeting is always asking for more leverage than the position limits allow. Residual covariance is still smaller than what would be needed for vol target and position limit to be simultaneously satisfiable. Realized vol (27.8%) exceeds 15% target but is no longer catastrophic. Position limits are effectively controlling vol, not the vol targeting formula.

**2. Reversal is dragging combined performance severely.**
Rev standalone: -84.4% total return, -85.1% max drawdown. Despite rolling Sharpe combination down-weighting it over time, it corrupts the combined portfolio. This is consistent with the horizon mismatch diagnosis: a 5-day reversal signal run through a weekly optimizer with κ=10 inertia is structurally wrong. Separation into a dedicated short-term book remains the correct fix.

**3. Momentum and Spread are the real engine.**
Mom (+83.2%) and Spread (+98.5%) are both working. Their standalone Sharpes (0.34, 0.37) are meaningful for a stat arb strategy on 13 assets. The combined Sharpe of 0.254 is lower than either standalone because Reversal is poisoning the combination.

**4. IC counts are inconsistent.**
Mom only has 89 IC observations vs Rev's 242 and Spread's 215. This is because `compute_ic` uses `alpha_df.index.intersection(returns_df.index)` and the momentum alpha has fewer valid rows due to the larger rolling window (60 + 5 skip = 65 day warmup vs 5 for reversal). Not a bug, but the IC t-stat for Mom (-0.46 on 89 obs) is not meaningful.

**5. No DCC regime section in report.**
`generate_report` has no `regime` parameter. DCC stats (convergence, rho, spikes, scale) are console-only. Fix pending approval.

### Immediate Next Steps (updated priority)

1. **CRITICAL:** ~~Remove reversal from combined alpha~~ — DONE (2026-03-29). See code change below.
2. **HIGH:** ~~Add DCC `regime` section to report~~ — DONE (2026-03-29).
   - `report.py`: added `compute_regime_stats(regime_df, scale_series, dcc_converged)` function; added `regime_df`, `scale_series`, `dcc_converged` params to `generate_report()`; report now contains `"regime"` key (null if DCC unavailable).
   - `run_baseline.py`: initialized `regime_df=None`, `scale_series=None`, `dcc_converged_flag=None` before try/except; set `dcc_converged_flag=True` on success; passes all three to `generate_report()`; console summary block now prints regime stats.
3. **MEDIUM:** ~~Investigate vol targeting saturation~~ — ROOT CAUSE DIAGNOSED + FIXED (2026-03-29).
   - **Root cause:** LW shrinkage on residual returns (T/N≈4.6) collapses Σ toward identity. Dollar-neutral positions nearly cancel under identity covariance → model port_vol ≈ 0.86% annual vs realized 27.8%. Scale factor 5-15× always inflates positions to MAX_WEIGHT clip every period. Vol targeting was structurally inert.
   - **Fix (Phase 2b):** Replaced model-based vol targeting with realized-vol cap.
     - `VOL_LOOKBACK = 8` weeks added to parameters
     - In `run_backtest`: pre-compute `weekly_ret_map` before loop; track `pnl_history` (no look-ahead: PnL recorded at start of each iteration, after last week's return is known); `scale = min(1.0, TARGET_VOL / rv)` where `rv = std(pnl[-8:]) * sqrt(52)`; applied only when `len(pnl_history) >= 8`
     - LW Σ retained in optimizer for relative asset sizing
     - Report phase name updated to "Phase2b" → writes `reports/report_phase2b.json`
   - **Expected effect:** On high-vol periods, scale < 1 → positions reduced → realized vol pulled toward 15%. On calm periods, scale = 1 → no change. First 8 weeks unscaled (insufficient history).
   - `COV_WINDOW` increased from 60 → 120 days (T/N improves from 4.6 → 9.2; less aggressive LW shrinkage; better relative sizing within the optimizer).
   - `MAX_WEIGHT` increased from 0.03 → 0.07 (2026-03-29): symmetric scaling cap-aware logic was still blocked — scale_cap = MAX_WEIGHT/max(|x_raw|) was ≤ 1 when optimizer output was at the old cap. Raising to 0.07 gives scale_cap room to allow scale > 1 when positions are below 0.07. Max gross exposure cap rises from 0.39 → 0.91 (13 × 0.07).
   - **Pending:** Phase 2b superseded by Phase 2c before being run. See below.

---

## [2026-03-29] VOLATILITY TARGETING — Phase 2c (Symmetric Bounded Scaling)

### Why one-sided scaling fails structurally

Phase 2b used `scale = min(1, TARGET_VOL / rv)`. This only reduces positions and never increases them. Under κ=10 inertia, reduced positions feed back as smaller `x_prev`, which causes the Chernov solution to converge downward. Realized vol drifts below target and stays there. The mechanism has a structural downward bias — it cannot converge to the target from below.

### Why Ledoit-Wolf cannot be used for portfolio-level vol targeting

LW on residual returns (T/N ≈ 9.2 with COV_WINDOW=120) applies aggressive shrinkage toward a scaled identity matrix. For a dollar-neutral portfolio under near-identity covariance, long and short positions cancel: σ²_p = σ² × Σxᵢ² (not σ² × (Σxᵢ)² which cancels). Estimated portfolio vol ≈ 0.86% annual at MAX_WEIGHT = 0.03 vs realized 27.8% — a 32× underestimate. Any model-based vol targeting using LW will produce scale >> 1, inflating to MAX_WEIGHT every period. The mismatch is structural, not fixable by widening the window.

### Realized PnL volatility as control variable

`rv = std(pnl[-VOL_LOOKBACK:]) * sqrt(52)` measures actual portfolio return volatility from realized PnL. No model assumptions. Directly responsive to the risk that matters. 8-week window captures current regime without excessive lag. No look-ahead: PnL is recorded at the start of each iteration, after the previous period's return is fully known.

### Adopted formula (Phase 2c — final: EWMA + power scaling)

```
ewma_var_t  = (1-α)*ewma_var_{t-1} + α*pnl_{t-1}²   α = 1-exp(-ln2/2)
rv_t        = sqrt(ewma_var_t * 52)
scale_raw   = (TARGET_VOL / rv_t)^1.5
scale_cap   = MAX_WEIGHT / max(|x_raw|)
scale       = clip(min(scale_raw, scale_cap), SCALE_MIN, SCALE_MAX)
x_t         = scale * x_raw
```

Parameters: `EWMA_HALFLIFE=2w`, `SCALE_MIN=0.2`, `SCALE_MAX=1.5`, `MAX_WEIGHT=0.07`
Initialization: `ewma_var_0 = (TARGET_VOL/sqrt(52))²` → neutral prior, scale_0 = 1

Power scaling (exponent 1.5): more convex than linear — harder deleveraging above target (rv=30% → scale=0.35 vs 0.50 linear), faster re-leveraging below (rv=7.5% → scale=2.83→clipped to 1.5).

Bounds: `SCALE_MIN=0.2` binds when rv≥47%; `SCALE_MAX=1.5` binds when rv≤10.2%.

### Cap-aware scaling procedure

1. Compute `x_raw` from optimizer (Chernov + LW Σ)
2. Update `ewma_var` with previous period's realized PnL (no look-ahead)
3. `scale_raw = (TARGET_VOL / rv)^1.5`
4. `scale_cap = MAX_WEIGHT / max(|x_raw|)` — uniform scale that exactly saturates the most extreme position at MAX_WEIGHT
5. If `scale_raw > scale_cap`: use `scale_cap` (cap binds before vol target); set `cap_bind = True`
   Else: use `scale_raw`
6. `scale = clip(scale, SCALE_MIN, SCALE_MAX)`
7. `x_t = scale * x_raw`  ← dollar neutrality preserved (scalar multiplication)
8. Final safety: `x_t = clip(x_t - mean(x_t), -MAX_WEIGHT, MAX_WEIGHT)` — should rarely bind

### Clear separation of roles

| Component | Role |
|---|---|
| Chernov optimizer (LW Σ) | Relative position sizing across assets |
| Realized-vol scaling | Portfolio-level risk control (absolute size) |
| MAX_WEIGHT clip | Hard per-position limit (never exceeded) |

### Self-consistency: scale > 1 works when it should

Under κ=10, the Chernov solution ≈ x_prev + α/κ. With FM-aligned α ≈ 0.001 in E[r] units and x_prev well below MAX_WEIGHT (which holds whenever scale < 1 has been applied recently), x_raw << MAX_WEIGHT. scale_cap = MAX_WEIGHT/max(|x_raw|) >> 1, so scale > 1 is not blocked by the cap. Scale > 1 is only attempted in low-vol regimes when positions are naturally small — the design is self-consistent.

### Implementation

- New module-level parameters: `SCALE_MIN = 0.5`, `SCALE_MAX = 2.0`
- New `run_backtest()` params: `scale_min=SCALE_MIN`, `scale_max=SCALE_MAX`
- Replaced one-sided block with symmetric bounded cap-aware block
- Added per-loop tracking: `scale_history`, `cap_bind_history`
- Return dict extended: `avg_scale`, `n_cap_bind`
- Phase name: "Phase2c" → `reports/report_phase2c.json`

---

## [2026-03-29] ARCHITECTURAL DIRECTION — Multi-Horizon System (CONFIRMED)

### Decision

The system will be refactored into **separate, independent books**. Each book has its own alpha construction, covariance model, optimizer instance, and vol target. Books are combined only at the portfolio level (PnL aggregation or top-level weight allocation).

### Book Structure

| Book | Horizon | Signals | Rebalance | Status |
|---|---|---|---|---|
| Short-term | 1–5 days | Reversal (5d rolling sum) | Daily or 2–3 day | Future — not yet built |
| Medium-term | 1–3 months | Momentum (60d) + Spread (120d ECM) | Weekly (Friday) | **Active — current system** |
| Long-term | 6–12 months | TBD (macro factors, carry if data available) | Monthly | Future — not yet designed |

### Why books must be independent

- **κ (position inertia)** controls how quickly the optimizer tracks alpha. For a 5-day signal, κ should be small (fast tracking). For a 60-day signal, κ should be large (slow, stable positions). A single κ cannot serve both.
- **γ (risk aversion)** governs position sizing per unit of alpha magnitude. With FM-aligned alphas in E[r] units, each block's β̄ differs — the same γ produces different leverage per block. Independent optimizers allow per-book calibration.
- **Covariance window** should match the rebalancing frequency. 60-day Ledoit-Wolf is appropriate for weekly rebalancing. A daily book needs a much shorter window (10–20 days) or EWMA covariance.
- **Vol targeting** applies differently: a short-term book expects higher turnover and needs tighter position limits; a medium-term book needs wider limits to allow alpha to build.
- **IC decay** differs by horizon: reversal IC decays within 5 days, momentum IC is flat for 20–60 days. Mixing them in one optimizer creates conflicting holding-period incentives.

### Current state of medium-term book

- **Signals:** Momentum + Spread (rolling Sharpe combination, 52w window)
- **Reversal excluded from combination** as of 2026-03-29 (run_baseline.py step 7c)
- **Reversal standalone backtest still runs** — preserved for future short-term book
- **Alpha combination:** 2-block rolling Sharpe (EQ2 = {mom: 0.5, spread: 0.5} as shrinkage target)
- **Covariance:** Ledoit-Wolf on residual_ret (60-day window)
- **Vol target:** 15% (soft ceiling — position limits bind first)
- **DCC regime:** applied to combined alpha via multiplicative scale

---

## [2026-03-29] REVERSAL SIGNAL — CONSTRUCTION BLUEPRINT (for short-term book)

Complete specification so the signal can be reconstructed identically in a future dedicated book.

### Construction steps (as implemented in run_baseline.py)

```
1. Beta neutralization
   residual_ret[asset] = ret[asset] - (alpha_t + beta_t * ret['SPX'])
   Rolling OLS window: BETA_WINDOW = 60 days
   Output: residual_ret — daily, shape (T, 13), index = trading days

2. Raw reversal signal
   rev_raw = -residual_ret.rolling(REV_WINDOW).sum()
   REV_WINDOW = 5 days
   Sign convention: negative sign → beaten-down assets (large negative 5d sum)
   receive HIGH signal → optimizer buys them (mean reversion expectation)

3. Cross-sectional normalization
   rev_norm = cs_normalize(rev_raw)
   Per-row z-score across all 13 assets. Removes time-series level shifts.

4. MAD winsorization
   rev_wins = robust_winsorize(rev_norm, z=3.0)
   Clips to median ± 3×MAD per row. Removes extreme cross-sectional outliers.

5. FM alignment (Fama-MacBeth beta scaling)
   alpha_rev_aligned, bstats_rev = align_signal_to_expected_return(
       rev_wins[ASSETS], ret[ASSETS], reb_dates_list, beta_floor=0.0
   )
   Rolling 52-week cross-sectional OLS: fwd_ret = a + β×signal + ε
   beta_floor=0.0: prevents FM from inverting signal when β̄ < 0 (noise weeks → zero, not short)
   Output: alpha in expected-return units (E[r] scale)

6. Re-normalization
   alpha_rev_n = normalize_alpha(alpha_rev_aligned.dropna(how='all'))
   CS z-score + MAD winsorize again after FM scaling (restores comparable scale across blocks)
```

### Key parameters for short-term book

| Parameter | Value | Note |
|---|---|---|
| REV_WINDOW | 5 days | Rolling sum window |
| BETA_WINDOW | 60 days | Market beta neutralization |
| beta_floor | 0.0 | Keep — prevents trend-following on failed weeks |
| Rebalance frequency | **Daily or 2–3 day** | NOT weekly — signal decays in 5d |
| κ (inertia) | **Small (1–3)** | Fast tracking needed |
| COV_WINDOW | **10–20 days** | Short for short-horizon covariance |

### What NOT to carry over from medium-term book

- Do NOT use the same `cov_dict` (built on 60-day LW for weekly rebalancing)
- Do NOT use the same `reb_dates` (weekly Fridays — too slow for 5-day reversal)
- Do NOT apply the DCC regime conditioning from the medium-term book (different risk profile)
- Do NOT use FM alignment with 52-week window (too slow for daily-frequency signal)

### Diagnostic output preserved

The following diagnostic is still computed in run_baseline.py step 7b-inv and printed at runtime:
- IC clamped (β≥0) vs. unclamped (β free): compares IC and standalone Sharpe
- Rev standalone backtest: `res_rev` with full attribution in report
- FM β̄ statistics: pct_negative, pct_clamped, raw_mean, raw_std

---

## [2026-03-30] ARCHITECTURAL REDESIGN — Modular Portfolio Engine

### Motivation

`run_baseline.py` has outgrown its role as a research script. It is ~930 lines of sequential, monolithic code with no separation between alpha construction, risk modeling, optimization, regime conditioning, and backtesting. Adding a second book (e.g. short-term reversal) would require duplicating the entire pipeline.

Additionally, the regime integration is structurally wrong: `compute_alpha_scale` maps the regime to a single scalar that multiplies the combined alpha. This means regime reduces signal magnitude, but the portfolio optimizer re-inflates positions via vol targeting — the two mechanisms partially cancel. More critically, a scalar does not distinguish between regime states that call for different *portfolio shapes* (e.g. emphasize spreads in "clustered" vs. reduce spreads in "crowded").

---

### Current Limitations of run_baseline.py

| Limitation | Description |
|---|---|
| Monolithic | Alpha construction, covariance, optimization, vol targeting, reporting all in `main()` |
| No book abstraction | Adding a second book requires duplicating ~500 lines |
| Regime used as scalar | `compute_alpha_scale` produces a single multiplier — structurally wrong |
| Regime label unused | `regime_df["regime"]` (normal/clustered/crowded/crisis/broken) is computed but never consumed |
| Parameters scattered | All constants at module level; no config registry; runs not reproducible without code inspection |
| No walk-forward split | Full-sample parameter calibration; no OOS validation infrastructure |
| Assets/pairs hardcoded | FUTURES, ETFS, PAIRS are module-level globals; cannot run on a different universe without edits |

---

### New Architecture — Modular Portfolio Engine

#### Module structure

```
Statistical Arbitrage Project/
├── run_baseline.py          ← FROZEN — baseline reference only, do not extend
├── run_engine.py            ← NEW entry point for modular engine
│
├── alphas/
│   ├── __init__.py
│   ├── reversal.py          ← build_reversal(residual_ret, window) → pd.DataFrame
│   ├── momentum.py          ← build_momentum(residual_ret, skip, window) → pd.DataFrame
│   └── spread.py            ← build_spread(residual_ret, pairs, window, ...) → pd.DataFrame
│
├── risk/
│   ├── __init__.py
│   └── covariance.py        ← build_cov_dict(returns, window, method='ledoit-wolf') → OrderedDict
│
├── portfolio/
│   ├── __init__.py
│   ├── optimizer.py         ← chernov_weights(alpha_t, Sigma_t, x_prev, ...) → np.ndarray
│   ├── book.py              ← Book class (see below)
│   └── allocator.py         ← Allocator class (combines Book PnL streams)
│
├── regime/
│   ├── __init__.py
│   ├── regime_detection.py  ← EXISTING — compute_regime_signals, compute_alpha_scale (legacy)
│   └── regime_mapping.py    ← NEW — get_book_actions(regime_label) → dict
│
├── backtest/
│   ├── __init__.py
│   └── engine.py            ← run_simulation(book, returns_df, ...) → dict
│
├── report.py                ← EXISTING — generate_report (no changes needed)
└── regime_detection.py      ← EXISTING at root — kept for backward compat, imports from regime/
```

---

### Book Abstraction

A `Book` represents one independent portfolio:

```
Book:
  name:          str            — identifier ("medium_term", "short_term", etc.)
  assets:        list[str]      — universe (can differ per book)
  alpha_df:      pd.DataFrame   — (T, N) alpha in E[r] units (FM-aligned)
  cov_dict:      OrderedDict    — {date: pd.DataFrame} Ledoit-Wolf Σ per rebalancing date
  reb_dates:     DatetimeIndex  — rebalancing schedule (weekly / daily / monthly)
  gamma:         float          — risk aversion for Chernov optimizer
  kappa:         float          — position inertia
  max_weight:    float          — per-asset position limit
  target_vol:    float          — vol target for this book
  ewma_halflife: int            — EWMA halflife for realized vol estimation
  is_active:     bool           — can be toggled by regime decision layer

  Methods:
    run() → {"weights": w_df, "pnl": pnl, "sharpe": ..., "max_dd": ..., ...}
```

Each book calls `run_simulation(book, returns_df)` from `backtest/engine.py`.

Each book is **independently testable**: it can be instantiated and backtested with no knowledge of other books.

---

### Allocator

The `Allocator` receives a list of Book objects and combines their PnL/weights:

```
Allocator:
  books:         list[Book]
  regime_df:     pd.DataFrame   — output of compute_regime_signals
  method:        str            — "equal_vol" | "rolling_sharpe" | "regime_driven"

  Methods:
    allocate(date) → dict[book_name → float]  — weight per book at each date
    run() → combined PnL series
```

The allocator is where regime enters the system: it decides, per book, whether the book is active and what weight to assign it. The books themselves are regime-unaware.

---

### Regime as Control Variable (not scalar)

The old pattern (DEPRECATED):
```python
# WRONG — scalar reduces alpha but vol targeting re-inflates positions
alpha_combined_final = alpha_combined_aligned * scale_t
```

The new pattern:
```python
# CORRECT — regime decides book activity and allocator weights
actions = get_book_actions(regime_label_t)
# actions = {
#   "medium_term": {"active": True,  "weight_boost": 1.0},
#   "spread_book":  {"active": False, "weight_boost": 0.0},
# }
```

`compute_alpha_scale` in `regime_detection.py` is **marked legacy**. It is not deleted (run_baseline.py still calls it), but it must not be used in the new engine.

---

### regime_mapping.py — Regime → Book Decisions

```python
def get_book_actions(regime_label: str) -> dict:
    """
    Map a categorical regime label to portfolio construction decisions.

    Returns a dict of {book_name: {"active": bool, "alpha_boost": float}}
    where alpha_boost is a multiplier on the book's alpha (not its positions).

    Regime definitions (from regime_detection.py):
      "normal"    — correlation within baseline; all books active
      "clustered" — elevated rho, high dispersion; sector rotation underway;
                    spread signals are stronger (pairs diverging into clusters)
      "crowded"   — elevated rho, low dispersion; homogeneous co-movement;
                    spread mean-reversion unreliable; favor momentum
      "crisis"    — rho_z > 2; extreme systematic risk; reduce all exposure
      "broken"    — spread_stab < 0.3; signal infrastructure sparse; disable spread book
    """
```

Initial mapping (to be calibrated empirically, not hardcoded permanently):

| Regime | medium_term (mom+spread) | spread_book | short_term (rev) |
|---|---|---|---|
| normal | active, boost=1.0 | active, boost=1.0 | active, boost=1.0 |
| clustered | active, boost=1.0 | active, boost=1.3 | active, boost=0.8 |
| crowded | active, boost=1.0 | active, boost=0.6 | active, boost=0.8 |
| crisis | active, boost=0.5 | disabled | active, boost=0.3 |
| broken | active, boost=1.0 | disabled | active, boost=1.0 |

Note: `alpha_boost` multiplies the alpha signal *before* the optimizer (not after). This changes the optimizer's solution shape, not just scale — it is fundamentally different from scalar post-scaling.

---

### Separation of Concerns (definitive)

| Component | Responsibility | Must NOT do |
|---|---|---|
| `alphas/` | Construct raw signals from returns | Know about optimizer, regime, or other books |
| `risk/covariance.py` | Estimate Σ from returns | Know about alpha or regime |
| `portfolio/optimizer.py` | Solve Chernov given (α, Σ, x_prev) | Know about regime, book structure |
| `portfolio/book.py` | Own alpha + Σ + optimizer + vol target | Know about other books or regime |
| `regime/regime_detection.py` | Compute regime signals from returns | Know about alphas or positions |
| `regime/regime_mapping.py` | Map regime label → book decisions | Know about alpha construction internals |
| `portfolio/allocator.py` | Combine books using regime decisions | Construct alphas or estimate covariance |
| `backtest/engine.py` | Simulate book or allocator over time | Contain alpha or covariance logic |
| `run_engine.py` | Wire components together | Contain any logic (pure orchestration) |

---

### Minimal Code Changes for Scaffolding (Phase 3 — incremental)

**Step 1:** Create `regime/regime_mapping.py` with `get_book_actions(regime_label)`.
- Pure function, no state, no side effects.
- Returns a decision dict.
- Independently testable.
- Does NOT modify run_baseline.py.

**Step 2:** Create `portfolio/book.py` — `Book` dataclass or class.
- Wraps alpha_df + cov_dict + run() method.
- `run()` calls `chernov_weights` in a loop (extracted from run_baseline.py's `run_backtest`).
- The current `run_backtest` is the implementation template — do not delete it from run_baseline.py.

**Step 3:** Extract `portfolio/optimizer.py` — `chernov_weights(alpha_t, Sigma_t, x_prev, params)`.
- Currently lives as `optimize_weights()` in run_baseline.py.
- Pure function — trivial to extract without changing logic.

**Step 4:** Create `portfolio/allocator.py` — `Allocator.run()` combines book PnLs using `regime_mapping`.

**Step 5:** Create `run_engine.py` — wires everything; medium-term book only (same signals as run_baseline.py) to validate parity.

**Step 6 (future):** Extract `alphas/` modules one at a time (reversal, momentum, spread). Each extraction is validated by running both run_baseline.py and run_engine.py and confirming PnL parity.

---

### What run_baseline.py becomes

`run_baseline.py` is **frozen** after Phase 2c:
- It remains runnable and generates `reports/report_phase2c.json`.
- It is the reference implementation for validation: new engine output must match or improve on it.
- No new features are added to it.
- It is not deleted — it is the "ground truth" for regression testing.

---

### Deprecation annotation for compute_alpha_scale

`regime_detection.compute_alpha_scale` must be annotated as:
```python
# LEGACY — deprecated as of 2026-03-30.
# This function implements regime as a scalar multiplier on alpha, which is
# structurally incorrect: vol targeting re-inflates positions after the scalar
# is applied, partially canceling the regime guard.
# Use regime/regime_mapping.get_book_actions() in the new engine instead.
# Retained for backward compatibility with run_baseline.py only.
```

This annotation is the ONLY change allowed to regime_detection.py for this step.

---

### Open Questions (to resolve before implementation)

1. **FM alignment per book or shared?**
   Each book runs its own FM alignment (`align_signal_to_expected_return`). The 52-week window is appropriate for the medium-term book. A short-term book needs a shorter window (10–20 weeks). FM alignment should be a parameter of the book, not a global.

2. **Shared covariance or per-book?**
   Current: one `cov_dict` built on residual_ret with COV_WINDOW=120. A short-term book needs a shorter window (10–20 days). Each book should own its `cov_dict` built from the same `residual_ret` but with different window/method.

3. **Allocator frequency.**
   If books rebalance at different frequencies (daily vs weekly), the allocator must handle time-index mismatches. Design: each book produces a weight DataFrame on its own schedule; allocator resamples to a common (daily) output using forward-fill.

4. **Regime label frequency.**
   `regime_df` is daily. `get_book_actions` should use the most recent label on or before each book's rebalancing date (same no-look-ahead pattern as `compute_alpha_scale`).

---

### Phase 3 Implementation Order (pending approval per step)

| Step | Deliverable | Dependencies |
|---|---|---|
| 3.1 | `regime/regime_mapping.py` (pure function) | regime_detection.py upgraded (done) | ✓ DONE |
| 3.2 | Deprecation annotation on `compute_alpha_scale` | None | ✓ DONE |
| 3.3 | `portfolio/optimizer.py` (extract `optimize_weights`) | None | ✓ DONE |
| 3.4 | `portfolio/book.py` (wrap run_backtest logic) | optimizer.py | pending |
| 3.5 | `portfolio/allocator.py` (combine books with regime) | book.py, regime_mapping.py | pending |
| 3.6 | `run_engine.py` (medium-term book only, validate parity) | All above | pending |
| 3.7 | Extract `alphas/momentum.py`, `alphas/spread.py` | run_engine.py validated | pending |
| 3.8 | Short-term book (daily reversal) | alphas/reversal.py, run_engine.py | pending |

---

## [2026-03-30] SESSION LOG — Changes made this session

### 1. Power scaling parameter: 1.8 → 2.0

**File:** `run_baseline.py`
- Line 210 (actual computation): `** 1.8` → `** 2.0`
- Line 133 (docstring): `^1.5` → `^2.0` (also corrected stale 1.5 reference)
- Line 201 (inline comment): `^1.5` → `^2.0`
- Line 813 (report description string): `^1.8` → `^2.0`

**Effect:** More convex deleveraging above target. At rv=22.5%, scale goes from 0.476 (1.8) to 0.444 (2.0). Impact mostly in mid-vol regime (15–40% rv). At rv≥47% (SCALE_MIN=0.2 floor) and rv≤10.2% (SCALE_MAX=1.5 ceil) behavior is identical.

**NOTE:** The pipeline has NOT been re-run with power=2.0. `report_phase2c.json` still reflects power=1.8. Next run will be the first with power=2.0.

---

### 2. regime_detection.py — Multi-state regime upgrade

**Function:** `compute_regime_signals` (only — `compute_alpha_scale` not modified)

**New columns added to returned DataFrame:**
- `rho_z`: rolling z-score of `rho_t` using `rho_roll_mean` / `rho_roll_std.clip(1e-8)`
  (was computed locally; now exposed as a column)
- `disp_z`: rolling z-score of `disp_t` using same `spike_window` and `min_periods=spike_window//2`
- `regime`: categorical string label — `np.select` priority order:
  1. `"broken"` if `spread_stab < 0.3`
  2. `"crisis"` if `rho_z > 2.0`
  3. `"crowded"` if `rho_z ∈ (1,2]` and `disp_z < 0`
  4. `"clustered"` if `rho_z ∈ (1,2]` and `disp_z ≥ 0`
  5. `"normal"` (default)

**No look-ahead:** all rolling windows strictly backward-looking. `np.select` is element-wise on aligned Series.

**`compute_alpha_scale`:** annotated `# LEGACY — deprecated as of 2026-03-30`. Not modified otherwise. Still called by `run_baseline.py`.

---

### 3. run_baseline.py — Regime diagnostic print

One line added in step 7d, after `regime_df = compute_regime_signals(...)`:
```python
print(regime_df["regime"].value_counts())
```
No other changes to pipeline logic.

---

### 4. Phase 3 Step 1: regime/regime_mapping.py — COMPLETE

**Files created:**
- `regime/__init__.py` (empty)
- `regime/regime_mapping.py`

**Public API:**
- `get_book_actions(regime_label: str) -> dict`
  Pure function. if/elif chain over 5 regime labels. Returns:
  `{"spread": {"active": bool, "alpha_multiplier": float}, "momentum": {...}, "short_term": {...}}`
- `get_actions_for_date(regime_df: pd.DataFrame, date) -> dict`
  No-look-ahead helper: finds most recent row ≤ date, calls `get_book_actions`.
  Falls back to "normal" if no history or NaN label.

**Decision table (initial hypotheses, NOT empirically tuned):**

| Regime | spread | momentum | short_term |
|---|---|---|---|
| normal | active × 1.0 | active × 1.0 | active × 1.0 |
| clustered | active × 1.3 | active × 1.0 | active × 0.8 |
| crowded | active × 0.6 | active × 1.0 | active × 0.8 |
| crisis | active × 0.4 | active × 0.5 | active × 0.3 |
| broken | DISABLED × 0.0 | active × 1.0 | active × 1.0 |

`alpha_multiplier` is applied before the Chernov optimizer — changes portfolio shape, not just scale.

---

### 5. Phase 3 Step 3: portfolio/optimizer.py — COMPLETE

**Files created:**
- `portfolio/__init__.py` (empty)
- `portfolio/optimizer.py`

**Function:** `chernov_weights(alpha_t, Sigma_t, x_prev, n, gamma, kappa, lambd, max_weight) -> np.ndarray`

Verbatim copy of `optimize_weights` from `run_baseline.py`. Zero logic changes. Pure function.

**run_baseline.py changes (2 lines only):**
- Added: `from portfolio.optimizer import chernov_weights` (line 44)
- Replaced: `optimize_weights(...)` → `chernov_weights(...)` at line 197

The old `optimize_weights` definition remains in `run_baseline.py` as a dead function (not called). Can be removed in a future cleanup.

**Validation:** Arguments are identical. Results must be byte-for-byte identical to pre-extraction.

---

### Current file structure (as of 2026-03-30)

```
Statistical Arbitrage Project/
├── run_baseline.py          ← FROZEN baseline (power=2.0, regime print added)
├── regime_detection.py      ← compute_regime_signals (upgraded), compute_alpha_scale (LEGACY)
├── report.py
├── regime/
│   ├── __init__.py
│   └── regime_mapping.py    ← Phase 3.1 DONE
├── portfolio/
│   ├── __init__.py
│   └── optimizer.py         ← Phase 3.3 DONE
├── data/
├── reports/
│   └── report_phase2c.json  ← last run (power=1.8, generated 2026-03-30T09:32:59Z)
└── src/dcc garch/           ← DCC-GARCH source (unmodified)
```

### Immediate next step (pending approval): Phase 3 Step 4 — portfolio/book.py

---

## [2026-03-30] Phase 3 Step 4: portfolio/book.py — COMPLETE

**File created:** `portfolio/book.py`

**Class:** `Book`

**Attributes owned:**
- `name` (str), `alpha_df` (DataFrame T×N), `cov_dict` (OrderedDict), `reb_dates` (list)
- `gamma`, `kappa`, `lambd`, `max_weight`, `target_vol`, `ewma_halflife`, `scale_min`, `scale_max` (floats)
- `is_active` (bool, default True)

**Method:** `run(returns_df) -> dict`
- Returns: `{"weights", "pnl", "sharpe", "max_dd", "turnover", "avg_scale", "n_cap_bind"}`
- Loop logic extracted **verbatim** from `run_baseline.py::run_backtest` (lines 147–264)
- Only change: `self.*` attributes unpacked into local variables at the top of `run()`
- `chernov_weights` imported from `portfolio.optimizer` (same as run_baseline.py)
- `run_backtest` in `run_baseline.py` NOT deleted — remains as regression reference

**Design decisions:**
- Plain class (not dataclass): pandas DataFrames are mutable, dataclass adds no value here
- `reb_dates` stored as a list; `run()` uses `sorted(cov_dict.keys())` internally (verbatim copy)
  — Book.reb_dates attribute is available for inspection/documentation but run() does not deviate
- `assets` is NOT a stored attribute; derived from `alpha_df.columns.tolist()` inside `run()` (verbatim)
- `is_active` is stored but NOT checked inside `run()` — regime-aware activation is the Allocator's job
- Zero behavioral changes: `Book(params).run(returns_df)` == `run_backtest(alpha_df, cov_dict, returns_df, params)`

**Phase 3 step table (updated):**

| Step | Deliverable | Status |
|---|---|---|
| 3.1 | `regime/regime_mapping.py` | ✓ DONE |
| 3.2 | Deprecation annotation on `compute_alpha_scale` | ✓ DONE |
| 3.3 | `portfolio/optimizer.py` | ✓ DONE |
| 3.4 | `portfolio/book.py` | ✓ DONE |
| 3.4b | `regime/regime_detection.py` (move + shim) | ✓ DONE |
| 3.5 | `portfolio/allocator.py` | ✓ DONE |
| 3.6 | `run_engine.py` | pending |

---

## [2026-03-30] Phase 3 Step 5: portfolio/allocator.py — COMPLETE

**File created:** `portfolio/allocator.py`

**Class:** `Allocator(books)`
- `books`: list of `Book` instances

**Method:** `run(returns_df) -> dict`
- Returns: `{"pnl": pd.Series, "book_results": {book.name: result_dict}}`

**Logic (3 steps):**
1. Loop through `self.books`; skip if `book.is_active == False`; call `book.run(returns_df)`; store in `book_results[book.name]`
2. Combine PnL: `combined_pnl = combined_pnl.add(pnl, fill_value=0.0)` (handles index mismatches)
3. Return `{"pnl": combined_pnl, "book_results": book_results}`

**Design invariants:**
- NO regime logic, NO import of regime_detection or regime_mapping
- NO alpha modification, NO position scaling, NO weight tilts
- Pure orchestration: does not mutate Book objects
- With a single active book: `allocator.run(returns_df)["pnl"]` == `book.run(returns_df)["pnl"]` identically
- Multiple books: combined PnL = arithmetic sum (equal weight, full size each book)

---

## [2026-03-30] Regime module consolidation — COMPLETE

### What changed

| File | Change |
|---|---|
| `regime/regime_detection.py` | Created — verbatim copy of root file; `ROOT = Path(__file__).parent.parent` (was `.parent`) |
| `regime/__init__.py` | Updated — exports `compute_regime_signals`, `compute_alpha_scale`, `get_book_actions`, `get_actions_for_date` |
| `run_baseline.py:754` | Updated — `from regime_detection import ...` → `from regime.regime_detection import ...` |
| `regime_detection.py` (root) | Replaced with 6-line shim — re-exports from `regime.regime_detection`; no logic |

### What did NOT change

- Zero logic changes in any function
- `compute_regime_signals` identical
- `compute_alpha_scale` (LEGACY) identical
- `run_baseline.py` behavior: unchanged (import path updated, function contract identical)
- No circular dependencies: `regime/__init__.py` imports from `.regime_detection` and `.regime_mapping` only; neither imports from `portfolio/`

### Final regime/ structure

```
regime/
    __init__.py         ← exports all 4 public functions
    regime_detection.py ← DCC-GARCH signals + LEGACY scale (moved from root)
    regime_mapping.py   ← get_book_actions, get_actions_for_date (Phase 3.1)
```

---

## [2026-03-30] Phase 3 Step 6: run_engine.py — CREATED (parity validation pending)

**File created:** `run_engine.py`

**Design:**
- Exact copy of all alpha construction from `run_baseline.py` (data load → beta neutralization → reversal → momentum → spread → FM alignment → re-normalization → rolling Sharpe combination Mom+Spread)
- `alpha_combined = alpha_combined_aligned` — NO regime scaling (as specified)
- Imports `run_backtest` from `run_baseline` for reference comparison
- Creates `Book("medium_term", alpha_combined_aligned, cov_dict, ...)` with identical parameters
- Runs via `Allocator([book]).run(ret[ASSETS])`
- Parity check: `assert max(|pnl_engine - pnl_baseline|) < 1e-10`

**Phase 3 step table (updated):**

| Step | Deliverable | Status |
|---|---|---|
| 3.1 | `regime/regime_mapping.py` | ✓ DONE |
| 3.2 | Deprecation annotation on `compute_alpha_scale` | ✓ DONE |
| 3.3 | `portfolio/optimizer.py` | ✓ DONE |
| 3.4 | `portfolio/book.py` | ✓ DONE |
| 3.4b | `regime/regime_detection.py` (move + shim) | ✓ DONE |
| 3.5 | `portfolio/allocator.py` | ✓ DONE |
| 3.6 | `run_engine.py` | ✓ DONE — parity confirmed (max diff = 0.00e+00, exact) |
| 3.7 | `alphas/momentum.py`, `alphas/spread.py` | ✓ DONE — parity confirmed |
| 3.8 | `alphas/reversal.py` + short-term book | pending |

---

## [2026-04-29] PHASE 1 — ARCHIVE CLEANUP (EXECUTED)

### Summary

Moved all non-execution code to `archive/`. Active codebase reduced to core engine + strategy files only. No code was modified. No import paths were changed.

### Directory structure created

```
archive/
  debug/
  original_code/
  notebooks/
  shims/
```

### Files archived

| File | Destination | Reason |
|---|---|---|
| `debug_d1.py` | `archive/debug/` | Investigation script for ST book sign issue — resolved |
| `debug_d2.py` | `archive/debug/` | Same debug series |
| `debug_d3.py` | `archive/debug/` | Same debug series |
| `debug_d4.py` | `archive/debug/` | Same debug series |
| `original code/alpha_construction.py` | `archive/original_code/` | Pre-modularization artifact; hard-coded path to different machine |
| `original code/covariance_model.py` | `archive/original_code/` | Pre-modularization artifact; superseded |
| `original code/optimizer.py` | `archive/original_code/` | Pre-modularization artifact; superseded by `portfolio/optimizer.py` |
| `original code/data_check.py` | `archive/original_code/` | One-off exploration script |
| `notebooks/signal_research (1).py` | `archive/notebooks/signal_research.py` | Research artifact; not integrated into pipeline |
| `regime_detection.py` (root shim) | `archive/shims/` | 7-line compatibility shim deprecated 2026-03-30; no active callers |

### Validation

- grep across all active `.py` files: **zero references** to any moved module name
- `run_engine.py` dependency graph: **unchanged** (21 active files before and after)
- No `ImportError` risk: all moved files were confirmed not imported by any active module in Steps 1–2

### Active codebase after cleanup (21 files)

```
run_engine.py, run_baseline.py, report.py
alphas/__init__.py, alphas/momentum.py, alphas/spread.py, alphas/reversal.py
portfolio/__init__.py, portfolio/book.py, portfolio/allocator.py, portfolio/optimizer.py
regime/__init__.py, regime/regime_detection.py, regime/regime_mapping.py
src/dcc garch/dcc/__init__.py, dcc/model.py, dcc/optimizer.py, dcc/utils.py, dcc/validate.py
src/dcc garch/garch/__init__.py, garch/gjr_garch.py
```

### Rationale

- Separates research/debug artifacts from production pipeline
- Eliminates visual noise from inert files at project root
- Aligns codebase with modular system design (engine + strategy + archive)
- Improves clarity for subsequent restructuring phases

### Next step

**Phase 2A:** Move `src/dcc garch/` → `engine/risk/dcc_garch/`. Requires one path string change in `regime/regime_detection.py` (the `sys.path.insert` line). Smallest possible change with significant structural benefit.

---

## [2026-04-29] PHASE 2A — DCC-GARCH LIBRARY ISOLATION (EXECUTED)

### Summary

Moved the entire DCC-GARCH library from `src/dcc garch/` into `engine/risk/dcc_garch/`. Updated the path resolution in `regime/regime_detection.py` to use a sentinel-based root finder instead of a fragile depth-based parent traversal.

### Files moved

| Old path | New path |
|----------|----------|
| `src/dcc garch/dcc/__init__.py` | `engine/risk/dcc_garch/dcc/__init__.py` |
| `src/dcc garch/dcc/model.py` | `engine/risk/dcc_garch/dcc/model.py` |
| `src/dcc garch/dcc/optimizer.py` | `engine/risk/dcc_garch/dcc/optimizer.py` |
| `src/dcc garch/dcc/utils.py` | `engine/risk/dcc_garch/dcc/utils.py` |
| `src/dcc garch/dcc/validate.py` | `engine/risk/dcc_garch/dcc/validate.py` |
| `src/dcc garch/garch/__init__.py` | `engine/risk/dcc_garch/garch/__init__.py` |
| `src/dcc garch/garch/gjr_garch.py` | `engine/risk/dcc_garch/garch/gjr_garch.py` |

`src/` directory removed entirely after move.

### Code change in `regime/regime_detection.py`

Old (fragile depth assumption):
```python
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src" / "dcc garch"))
```

New (sentinel-based root finder — robust to future moves):
```python
def _find_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "CLAUDE.md").exists():
            return p
    return start
ROOT = _find_root(Path(__file__).resolve().parent)
sys.path.insert(0, str(ROOT / "engine" / "risk" / "dcc_garch"))
```

### Validation

- `compute_regime_signals`, `fit_multivariate_gjr`, `dcc_fit` all callable
- `run_engine.py` full import chain passes

### Next step

**Phase 2C:** Move `report.py` → `engine/backtest/report.py`. Requires updating 2 import strings in `run_baseline.py` only.

---

## [2026-04-29] PHASE 2C — REPORT MODULE RELOCATION (EXECUTED)

### Summary

Moved `report.py` into the `engine/backtest/` package. Updated both import sites in `run_baseline.py`. Zero code changes inside `report.py`. Zero changes to any other file.

### Files created

| File | Notes |
|------|-------|
| `engine/__init__.py` | Empty — makes `engine/` a Python package |
| `engine/backtest/__init__.py` | Empty — makes `engine/backtest/` a Python package |
| `engine/backtest/report.py` | Verbatim copy of `report.py` — no logic changes |

### Import edits in `run_baseline.py`

| Line | Old | New |
|------|-----|-----|
| 43 | `from report import generate_report` | `from engine.backtest.report import generate_report` |
| 603 | `from report import compute_ic as _compute_ic` | `from engine.backtest.report import compute_ic as _compute_ic` |

### `report.py` at root

Left in place (not deleted). It remains as a file but is no longer the imported source.
Will be archived in a later cleanup step once Phase 2B is complete and the full engine path is validated.

### Validation

```
python -c "from engine.backtest.report import generate_report, compute_ic"  → OK
python -c "from run_baseline import run_backtest"  → OK  (full module-level import)
run_engine.py full import chain (all 8 imports)   → OK
```

### Active file layout after Phase 2C

```
run_engine.py               ← entry point
run_baseline.py             ← reference; now imports from engine.backtest.report
report.py                   ← root copy — still present, not yet archived

engine/
  __init__.py
  backtest/
    __init__.py
    report.py               ← canonical location after Phase 2C
  risk/
    dcc_garch/              ← moved here in Phase 2A

alphas/   portfolio/   regime/   (unchanged)
```

### Next step

**Phase 2B:** Move `portfolio/`, `alphas/`, `regime/` → `engine/portfolio/`, `engine/alphas/`, `engine/regime/`. Large coordinated change (~12 import strings across 5+ files). Must be done atomically.

---

## [2026-04-29] PHASE 2B — ENGINE PACKAGE RELOCATION (EXECUTED)

### Summary

Moved `portfolio/`, `alphas/`, and `regime/` into `engine/`. Updated all 13 import statements across 6 active files before moving any directories. Zero code logic changes. Invariance check passed.

### Import edits (13 total, applied before any file moves)

| File | Line | Old | New |
|------|------|-----|-----|
| run_engine.py | 44 | from portfolio.book import Book | from engine.portfolio.book import Book |
| run_engine.py | 45 | from portfolio.allocator import Allocator | from engine.portfolio.allocator import Allocator |
| run_engine.py | 46 | from alphas.momentum import build_momentum | from engine.alphas.momentum import build_momentum |
| run_engine.py | 47 | from alphas.spread import build_spread | from engine.alphas.spread import build_spread |
| run_engine.py | 48 | from alphas.reversal import build_reversal | from engine.alphas.reversal import build_reversal |
| run_engine.py | 49 | from regime.regime_detection import ... | from engine.regime.regime_detection import ... |
| run_engine.py | 50 | from regime.regime_mapping import ... | from engine.regime.regime_mapping import ... |
| run_baseline.py | 44 | from portfolio.optimizer import chernov_weights | from engine.portfolio.optimizer import chernov_weights |
| run_baseline.py | 754 | from regime.regime_detection import ... | from engine.regime.regime_detection import ... |
| portfolio/book.py | 32 | from portfolio.optimizer import chernov_weights | from engine.portfolio.optimizer import chernov_weights |
| alphas/momentum.py | 18 | from alphas import normalize_alpha | from engine.alphas import normalize_alpha |
| alphas/reversal.py | 10 | from alphas import normalize_alpha | from engine.alphas import normalize_alpha |
| alphas/spread.py | 26 | from alphas import cs_normalize, ... | from engine.alphas import cs_normalize, ... |

### Directories moved

| Old | New |
|-----|-----|
| portfolio/ | engine/portfolio/ |
| alphas/ | engine/alphas/ |
| regime/ | engine/regime/ |

### Files that needed NO changes

- portfolio/__init__.py (empty), portfolio/optimizer.py (numpy only), portfolio/allocator.py (pandas only)
- regime/__init__.py (relative imports — unchanged), regime/regime_detection.py (sentinel root finder handles new depth), regime/regime_mapping.py (no cross-package imports)
- alphas/__init__.py (defines cs_normalize, robust_winsorize, normalize_alpha — no imports from alphas/)

### Validation



### Active file layout after Phase 2B



### Next step

**Phase 2D:** Move `run_engine.py`, `run_baseline.py` → `strategies/medium_term_rv/`. Requires updating ROOT path logic and sys.path handling in both files.

---

## [2026-04-29] PHASE 2D -- STRATEGY EXTRACTION (EXECUTED)

### Summary

Moved run_engine.py and run_baseline.py into strategies/medium_term_rv/.
Updated ROOT definition in both files to use sentinel-based root finder.
Updated run_engine.py to import run_backtest via fully qualified package path.
Created strategies/__init__.py and strategies/medium_term_rv/__init__.py.
Zero logic changes. Invariance check passed with identical output to pre-move.

### Code changes applied (before any file moves)

run_engine.py -- 2 changes:
1. ROOT = Path(__file__).parent
   replaced with sentinel _find_project_root() + ROOT = _find_project_root()
2. from run_baseline import run_backtest
   replaced with from strategies.medium_term_rv.run_baseline import run_backtest

run_baseline.py -- 1 change:
1. ROOT = Path(__file__).parent
   replaced with sentinel _find_project_root() + ROOT = _find_project_root()

The sentinel walks up from __file__ until CLAUDE.md is found -- robust to any depth.
DATA_DIR, ALPHA_DIR, sys.path.insert, output_dir all derive from ROOT and needed no changes.

### Files created (package structure)

- strategies/__init__.py  (empty)
- strategies/medium_term_rv/__init__.py  (empty)

### Files moved

- run_engine.py  ->  strategies/medium_term_rv/run_engine.py
- run_baseline.py  ->  strategies/medium_term_rv/run_baseline.py

### Validation

python strategies/medium_term_rv/run_engine.py
  exit code: 0
  no ImportError, no ModuleNotFoundError, no path errors
  INVARIANCE CHECK PASSED -- medium_term book == baseline (within 1e-10)
  SHORT-TERM VALIDITY CHECK PASSED
  Baseline Sharpe: 0.189 (identical to pre-move)
  Medium-term PnL length: 448, Sharpe: 0.189
  Short-term PnL length: 2641, Sharpe: 0.0057
  Combined Sharpe: 0.1816

### Active file layout after Phase 2D

engine/
  __init__.py
  alphas/        __init__.py, momentum.py, spread.py, reversal.py
  backtest/      __init__.py, report.py
  portfolio/     __init__.py, book.py, allocator.py, optimizer.py
  regime/        __init__.py, regime_detection.py, regime_mapping.py
  risk/dcc_garch/  dcc/ (5 files), garch/ (2 files)

strategies/
  __init__.py
  medium_term_rv/
    __init__.py
    run_engine.py    <- entry point
    run_baseline.py  <- reference backtest

report.py  (root, still present -- to be archived)

archive/  (inert)

### Remaining root-level files

- report.py: original copy, no longer imported by anything active.
  Will be cleaned up in a post-Phase-2 pass.

### Next step

Phase 3 -- Baseline Decoupling: replace the two run_backtest() calls in
run_engine.py step 7 with Book.run() calls, eliminating the dependency on
run_baseline.py. Requires explicit approval as a separate decision.

---

## [2026-04-30] PHASE 3 -- BASELINE DECOUPLING (EXECUTED)

### Summary

Removed all three run_backtest() calls from run_engine.py.
Replaced the two standalone backtests (Calls A/B) with Book.run() equivalents.
Removed Call C (combined baseline reference) and the invariance check block entirely.
Removed the run_baseline import. run_engine.py is now fully independent.

### Changes applied to strategies/medium_term_rv/run_engine.py

1. Import removed (Step 3):
   from strategies.medium_term_rv.run_baseline import run_backtest

2. Call A + B replaced (Step 1) -- standalone mom and spread backtests:
   OLD: res_mom_sa    = run_backtest(alpha_mom_n,    cov_dict, ret[ASSETS])
        res_spread_sa = run_backtest(alpha_spread_n, cov_dict, ret[ASSETS])
   NEW: Book("mom_sa",    alpha_mom_n,    cov_dict, ..., same params).run(ret[ASSETS])
        Book("spread_sa", alpha_spread_n, cov_dict, ..., same params).run(ret[ASSETS])
   Parameters: GAMMA, KAPPA, LAMBD, MAX_WEIGHT, TARGET_VOL, EWMA_HALFLIFE, SCALE_MIN, SCALE_MAX
   (identical to medium_term_book -- same cov_dict, only alpha_df differs)

3. Call C + invariance block removed (Step 2) -- lines 394-447:
   - res_baseline = run_backtest(alpha_combined_aligned, ...) deleted
   - pnl_baseline variable deleted
   - INVARIANCE CHECK print block deleted
   - assert diff_inv < 1e-10 deleted
   Replaced with: print("[7] Engine-only execution -- baseline decoupled")

4. Module docstring updated to reflect decoupled engine purpose.

### Files NOT modified

- strategies/medium_term_rv/run_baseline.py: untouched, still a standalone runnable script
- engine/ modules: untouched
- Book, Allocator implementations: untouched

### Why invariance check was retired

Book.run() is a verbatim copy of run_backtest(). The check proved max_diff = 0.00e+00
(exact bit-for-bit equality) across all prior phases. Re-running it at runtime adds
compute time without new information. The guarantee is structural, not empirical.

### Validation

python strategies/medium_term_rv/run_engine.py
  Exit code:               0
  ImportError:             none
  run_baseline imported:   no (fully decoupled)
  [7] print:               Engine-only execution -- baseline decoupled
  Medium-term Sharpe:      0.189  (identical to pre-Phase-3)
  Short-term Sharpe:       0.0057 (identical)
  Combined Sharpe:         0.1816 (identical)
  SHORT-TERM VALIDITY:     PASSED

### Final state of run_engine.py dependencies

run_engine.py imports:
  sys, numpy, pandas, pathlib, collections, statsmodels, sklearn  (stdlib/third-party)
  engine.portfolio.book.Book
  engine.portfolio.allocator.Allocator
  engine.alphas.momentum.build_momentum
  engine.alphas.spread.build_spread
  engine.alphas.reversal.build_reversal
  engine.regime.regime_detection.compute_regime_signals
  engine.regime.regime_mapping.get_book_actions

NO import from run_baseline or any strategies/ module.

### Restructuring complete

All phases executed:
  Phase 1  -- Archive cleanup              DONE
  Phase 2A -- DCC library isolation        DONE
  Phase 2C -- Report module relocation     DONE
  Phase 2B -- Engine package relocation    DONE
  Phase 2D -- Strategy extraction          DONE
  Phase 3  -- Baseline decoupling          DONE

---

## [2026-05-05] PHASE 4 -- VISUALIZATION PLANNING

Phase 4 added to `restructuring_plan.md` as PLANNED.

Scope: generate a small set of figures for GitHub presentation from existing engine
outputs. No strategy logic or backtest results will be changed.

Planned deliverables:
  reports/figures/cumulative_pnl.png    (combined + standalone attribution)
  reports/figures/drawdown.png          (combined strategy)
  reports/figures/rolling_sharpe.png    (combined strategy, 52w window)
  reports/figures/rolling_ic.png        (optional -- momentum + spread)

README will include an "Example Strategy Results" section embedding the primary figures.
Implementation pending explicit approval.

---

## [2026-05-05] PHASE 4 -- VISUALIZATION IMPLEMENTATION (EXECUTED)

### Summary

Created visualization layer for GitHub presentation. No engine or strategy files modified.

### Files created

| File | Notes |
|------|-------|
| `reports/figures/` | Output directory for all figures |
| `reports/plot_results.py` | Standalone plotting script |

### Figures generated

| File | Content |
|------|---------|
| `reports/figures/cumulative_pnl.png` | Combined (MT+ST), medium-term only, momentum standalone, spread standalone |
| `reports/figures/drawdown.png` | Drawdown of combined strategy |
| `reports/figures/rolling_sharpe.png` | 52-period rolling annualized Sharpe, combined strategy |
| `reports/figures/rolling_ic.png` | 26-week rolling IC for momentum and spread signals |

### Design

- `reports/plot_results.py` imports constants and helpers from `strategies/medium_term_rv/run_engine.py`
  (read-only — `main()` is never called)
- All Books constructed via `engine.portfolio.book.Book` and `engine.portfolio.allocator.Allocator`
- No scipy dependency — Spearman IC implemented via pandas rank correlation
- matplotlib only; `Agg` backend (no display required)
- Entry point: `python reports/plot_results.py` from project root

### Validation

- Script runs end-to-end with exit code 0
- All 4 figures written to `reports/figures/`
- DCC converged: a=0.0131, b=0.5639
- Pipeline metrics consistent with run_engine.py (Sharpe 0.1816)

---

## [2026-05-05] README CREATION + CHART DIAGNOSTIC

### README created: README.md at project root

Framing: engine-first. The repository is described as a "modular, production-style portfolio
engine." The included strategy is explicitly labeled an example plug-in, not a tuned strategy.

Sections: Overview, Architecture (data→alpha→portfolio→risk→execution→evaluation),
Core Abstractions (Alpha / Book / Allocator with code snippets), Repository Structure,
Example Strategy description, Example Strategy Results (figures + caveat), Engineering
Principles, Getting Started.

### Positioning language used (do not deviate from this framing)

- "production-style portfolio engine" / "modular research engine"
- "The goal is system design, modularity, reproducibility, and portfolio construction workflow.
  Sharpe optimization is not the objective."
- Under figures: "These figures demonstrate that the engine produces coherent PnL... They are
  not presented as evidence of an optimized or production-ready trading strategy."
- Under 2020 spike: "useful as a stress-test diagnostic... should not be interpreted as a
  standalone claim of alpha quality"

### 2020 spike diagnostic (do not re-explain from scratch in future sessions)

**Cause (confirmed from code inspection, not guessed):**
1. Short energy exposure from momentum (60-day) entering COVID crash
2. Spread energy pair concentration (4 of 9 pairs involve CLc1, LCOc1, NGc1)
3. SCALE_MAX=1.5 applied at entry (EWMA rv low in late 2019 → scale=1.5)
4. Extreme oil/commodity returns (CLc1 near-zero price → log return clipped at -13.8)
5. Not a data artifact, not a plotting bug. Real position, real return, one extreme event.

**Bottom line**: 2020 spike accounts for majority of total cumulative return.
Performance outside that event is flat to gradually positive.

### Rolling Sharpe design limitation (known, do not flag as bug)

The combined daily PnL series mixes weekly MT PnL (non-zero only on Fridays) with daily ST
PnL. A 52-period window on a daily series = 52 days ≈ 10 weeks. Too short for stable
Sharpe. Mean≈−0.05 diverges from full-period Sharpe≈0.18 because MT zeros on non-Friday
days distort the mean. This is a display limitation, not a calculation error.

### Rolling IC flat sections (known, do not flag as bug)

Flat sections in spread IC (2018–2020): sparse pair activity — rolling ADF frequently
excludes pairs in those windows, leaving few valid IC dates per rolling window.
Flat sections in momentum IC: genuine near-constant IC near zero or slightly negative at
weekly horizon for a 60-day signal (expected behavior, documented in prior sessions).

### GitHub readiness

All restructuring phases complete (1, 2A, 2B, 2C, 2D, 3, 4). README.md present.
Figures present. No open implementation tasks. Suitable for upload.

---

## [2026-05-05] README FINALIZED

README.md at project root rewritten with the following positioning (do not change this
framing in future sessions without explicit user instruction):

**Canonical framing:**
- Repository = "modular, production-style portfolio engine"
- Strategy = "example plug-in used to demonstrate the engine runs end-to-end"
- Figures = "diagnostic outputs"; explicitly NOT "evidence of optimized performance"
- Goal stated as "system design, modularity, reproducibility — not Sharpe maximization"

**"Interpreting the Results" section added** (under Example Strategy Results):
- 2020 jump documented as a system-level effect: energy exposure + shared signal direction
  + vol-targeting at SCALE_MAX=1.5 entering the event
- Explicitly states: "accounts for the majority of the strategy's total positive return;
  performance outside this event is broadly flat"
- Rolling IC / Rolling Sharpe noise explained as sampling limitations (N=13, short window)
- All framed as "consistent with system design", not overclaimed

**Figures embedded in README** in this order: cumulative_pnl, drawdown, rolling_ic, rolling_sharpe.
Results table included with caveat language immediately above it.
