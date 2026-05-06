# Restructuring Plan — Cross-Asset Portfolio Engine

## Objective

Transform the existing research/backtest project into a **modular cross-asset portfolio engine** suitable for:

- systematic trading research
- multi-strategy portfolio construction
- GitHub / interview presentation

The restructuring separates:

- reusable infrastructure (**engine**)
- strategy-specific logic (**strategies**)
- inactive / experimental code (**archive**)

---

## Target Structure

```
engine/
  alphas/
  portfolio/
  regime/
  risk/
  backtest/

strategies/
  medium_term_rv/

archive/
```

---

## Design Principles

- separation of concerns
- modular alpha construction
- asset-agnostic portfolio layer
- reproducible execution
- no breaking changes during migration
- step-by-step validation after each phase

---

## Phase Overview

### Phase 1 — Archive Cleanup (COMPLETED)

**Objective:**
Remove inactive code from the active codebase.

**Actions:**
- Move debug scripts, notebooks, original code, and shim files into `archive/`

**Validation:**
- No active module imports archived files
- Active codebase unchanged (21 files)
- `run_engine.py` executes without modification

**Outcome:**
Clean separation between production code and research artifacts

---

### Phase 2A — DCC Library Isolation (COMPLETED)

**Objective:**
Move DCC-GARCH library into engine structure

**Actions:**
- Move:
  ```
  src/dcc garch/ → engine/risk/dcc_garch/
  ```
- Update path resolution in `regime/regime_detection.py`

**Risks:**
- Broken path injection (`sys.path.insert`)
- incorrect relative path resolution

**Validation:**
- `compute_regime_signals()` runs successfully
- DCC + GJR-GARCH functions callable

---

### Phase 2C — Report Module Relocation (COMPLETED)

**Objective:**
Move reporting utilities into backtest module

**Actions:**
- Move:
  ```
  report.py → engine/backtest/report.py
  ```
- Update imports in `run_baseline.py`

**Risks:**
- hidden transitive dependency (module-level import)

**Validation:**
- `run_engine.py` still imports `run_baseline.py` successfully
- reporting functions accessible

---

### Phase 2B — Engine Package Relocation (COMPLETED)

**Objective:**
Group all reusable infrastructure under `engine/`

**Actions:**
- Move:
  ```
  portfolio/ → engine/portfolio/
  alphas/ → engine/alphas/
  regime/ → engine/regime/
  ```
- Update all import paths

**Risks:**
- widespread import breakage
- inconsistent namespace resolution

**Validation:**
- full pipeline runs
- invariance check passes

---

### Phase 2D — Strategy Extraction (COMPLETED)

**Objective:**
Separate strategy logic from engine

**Actions:**
- Move:
  ```
  run_engine.py
  run_baseline.py
  → strategies/medium_term_rv/
  ```
- Update:
  - ROOT path logic
  - sys.path handling

**Risks:**
- broken data paths
- broken module resolution

**Validation:**
- strategy runs from new location
- identical outputs to pre-move

---

### Phase 3 — Baseline Decoupling (COMPLETED)

**Objective:**
Remove dependency on legacy monolithic backtest

**Actions:**
- Replace:
  ```
  run_backtest() → Book.run()
  ```
- Refactor alpha combination logic
- retire or isolate invariance check

**Risks:**
- breaking numerical equivalence
- subtle PnL differences

**Validation:**
- PnL consistency preserved
- engine runs independently of `run_baseline.py`

---

### Phase 4 — GitHub Presentation Visualizations (PLANNED)

**Objective:**
The engine is structurally complete. This phase adds a small set of clean, purposeful figures
that communicate performance, risk, attribution, and signal validation for the GitHub presentation.
No strategy logic, backtest results, or parameters are changed.

**Deliverables:**

1. Create output directory:
   ```
   reports/figures/
   ```

2. Generate and save the following figures:
   ```
   reports/figures/cumulative_pnl.png   (required)
   reports/figures/drawdown.png          (required)
   reports/figures/rolling_sharpe.png    (required)
   reports/figures/rolling_ic.png        (optional)
   ```

3. Recommended figure content:
   - **cumulative_pnl.png** — cumulative PnL for combined strategy, momentum standalone,
     and spread standalone; optionally include reversal as a labeled excluded-attribution line
   - **drawdown.png** — drawdown curve for the combined strategy
   - **rolling_sharpe.png** — rolling Sharpe (52-week window) for the combined strategy
   - **rolling_ic.png** — rolling IC for momentum and spread, if IC series is already
     available from existing report outputs or trivially computable

4. Design rules:
   - Minimal and professional — no dashboard sprawl
   - No unnecessary heatmaps or decorative subplots
   - Figures must support the README narrative, not replace it
   - All plots generated from reproducible report/backtest outputs (not manually edited)
   - Consistent style: single color palette, axis labels, no chartjunk

5. README integration plan:
   - Add a short "Example Strategy Results" section to README
   - Embed only the primary figures: cumulative PnL and drawdown
   - Caption should note that the included strategy is an example medium-term
     relative value strategy built on the reusable engine
   - README narrative should describe the engine first, the strategy second

**Risks:**
- None to strategy logic or backtest results (read-only use of existing outputs)
- Figures may need regeneration if strategy outputs change

**Validation:**
- Figure generation script runs from project root without error
- Output files written to `reports/figures/`
- All plots sourced from existing engine outputs (no new backtests)
- No changes to any `.py` file in `engine/` or `strategies/`
- No new optimization or parameter tuning

---

## Execution Strategy

The migration is strictly **phased and validated**:

1. isolate inactive code
2. move self-contained components
3. update imports incrementally
4. move strategy layer last
5. decouple legacy logic

No phase introduces:

- architectural redesign
- new features
- performance optimization

---

## End Goal

A clean, modular system:

- reusable engine
- pluggable strategies
- clear separation between research and production logic

The repository should present as:

> a cross-asset portfolio engine with an example statistical arbitrage strategy

—not as a backtest script.
