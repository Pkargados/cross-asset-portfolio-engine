# DCC-GARCH Python Library

A mathematically faithful Python implementation of the Dynamic Conditional Correlation (DCC) and Asymmetric DCC (ADCC) GARCH models, reverse-engineered from the R package `rmgarch` and validated against the original academic literature.

## References

- Engle, R. (2002). Dynamic Conditional Correlations: A Simple Class of Multivariate GARCH Models. Journal of Business & Economic Statistics, 20(3), 339-350.
- Cappiello, L., Engle, R., & Sheppard, K. (2006). Asymmetric Dynamics in the Correlations of Global Equity and Bond Returns. Journal of Financial Econometrics, 4(4), 537-572.

---

## Model

### DCC (Engle 2002)

The model operates in two stages. Stage 1 (univariate GARCH) is assumed pre-completed. This library handles Stage 2 only.

Standardized residuals:

z_{i,t} = eps_{i,t} / sigma_{i,t}

Q_t recursion:

Q_t = (1 - a - b) * Q_bar  +  a * z_{t-1} z_{t-1}'  +  b * Q_{t-1}

Correlation matrix:

R_t[i,j] = Q_t[i,j] / sqrt(Q_t[i,i] * Q_t[j,j])

Conditional covariance:

H_t = Sigma_t * R_t * Sigma_t

---

## Integration with raw returns (CRITICAL)

This library assumes standardized residuals (Z) and conditional volatilities (sigmas)
are provided as inputs.

To use this library starting from raw returns:

1. Fit univariate GARCH (GJR-GARCH) for each asset:
   → obtain eps_{i,t}, sigma_{i,t}

2. Construct standardized residuals:
   z_{i,t} = eps_{i,t} / sigma_{i,t}

3. Pass:
   Z, sigmas → DCC fit()

This library does NOT perform Stage 1 automatically.

---

## Intended use in this project (CRITICAL)

This DCC-GARCH implementation is used ONLY for:

- Regime detection
- Correlation monitoring

It is NOT used for:

- Portfolio covariance estimation
- Optimization inputs

Key outputs used:

1. Average correlation:
   rho_t = mean of off-diagonal elements of R_t

2. Correlation dispersion:
   std of off-diagonal elements

3. Correlation spikes:
   sudden increases in rho_t

These signals are used to adjust:

- Alpha weights
- Risk exposure

---

## Usage

import pickle
from python.dcc import fit

with open('data/dcc_inputs.pkl', 'rb') as f:
    data = pickle.load(f)

Z = data['Z']
sigmas = data['sigmas']

result = fit(Z, sigmas, model='DCC')

print(result['R'].shape)  # (T, N, N)

---

## Inputs

Z: (T, N) standardized residuals  
sigmas: (T, N) conditional volatilities  

---

## Outputs

R: (T, N, N) conditional correlation matrices  
H: (T, N, N) conditional covariance matrices  

---

## Notes

- DCC covariance (H_t) is NOT used in portfolio optimization.
- Ledoit-Wolf covariance is used instead for optimization.
