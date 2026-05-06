# Cross-Asset Portfolio Engine

This project is being refactored from a research/backtest codebase into a **modular trading system** for cross-asset statistical arbitrage.

The objective is to transform the current code into a **clean, GitHub-ready portfolio engine**, aligned with industry architecture.

---

# 0. EXECUTION PROTOCOL (MANDATORY)

You MUST follow strict step-by-step execution:

- Perform ONE step at a time
- After each step: STOP
- Wait for user approval before continuing
- Do NOT proceed autonomously
- Do NOT rewrite large parts of code without justification

After each step, end with:

"STEP COMPLETE. Waiting for user approval to proceed."

---

# 1. CONTEXT

The project currently contains:

- Working system:
  - run_engine.py
  - regime_detection.py

- Baseline / reference:
  - run_baseline.py

- Additional files:
  - may include legacy scripts, experiments, or unused modules

The goal is to:

- Identify the **true core engine**
- Separate it from legacy code
- Restructure the project into a **modular portfolio engine**

---

# 2. TARGET SYSTEM (REFERENCE ARCHITECTURE)

The final system must follow this structure:

data → alpha → portfolio → risk → execution → evaluation

Core components:

- Alpha modules (momentum, spread)
- Book abstraction (strategy container)
- Allocator (multi-book combination)
- Risk model (covariance, vol targeting)
- Regime module (behavioral adjustments)
- Backtest engine

This is a **portfolio engine**, not a single strategy.

---

# 3. STEP 1 — FULL CODEBASE AUDIT (MANDATORY)

You MUST:

## 3.1 Identify all files

- List all .py files in the project
- Categorize each as:
  - CORE (used in current system)
  - BASELINE (reference only)
  - LEGACY (unused or outdated)

## 3.2 Trace execution

- Read:
  - run_engine.py
  - regime_detection.py
  - run_baseline.py

- Identify:
  - entry point
  - data flow
  - dependencies
  - which modules are actually used

## 3.3 Output:

SECTION 1: File Classification  
SECTION 2: Execution Flow  
SECTION 3: Core vs Legacy Separation  

DO NOT modify any code.

STOP after audit.

---

# 4. STEP 2 — CORE ENGINE ISOLATION

Based on audit:

- Define the **minimal set of files** required to run:
  - the engine
  - the example strategy

- Remove (logically, not physically yet):
  - unused modules
  - redundant scripts

Output:

- List of files to KEEP
- List of files to ARCHIVE

DO NOT delete anything yet.

STOP.

---

# 5. STEP 3 — SYSTEM RESTRUCTURING (ARCHITECTURE)

Reorganize the project into:

engine/
  alphas/
  portfolio/
  risk/
  regime/
  backtest/

strategies/
  medium_term_rv/

archive/

Rules:

- Move code WITHOUT changing logic
- Preserve functionality
- Do NOT refactor internals unless necessary

Map existing files into this structure.

Output:

- Proposed new folder structure
- Mapping: old file → new location

STOP.

---

# 6. STEP 4 — STRATEGY EXTRACTION

The current implementation (momentum + spread) must become:

strategies/medium_term_rv/

This includes:

- signal construction
- parameter choices
- configuration

The engine must NOT depend on this strategy.

Goal:

- Engine = reusable
- Strategy = plug-in example

STOP.

---

# 7. STEP 5 — ENGINE CLEANUP

Within engine:

- Ensure separation between:
  - alpha construction
  - portfolio logic
  - risk logic
  - regime logic

- Remove:
  - hardcoded paths
  - strategy-specific assumptions

- Ensure:

  alpha → expected return → optimizer → weights

DO NOT introduce new features.

Only clean structure.

STOP.

---

# 8. STEP 6 — README PREPARATION

Prepare content for README:

- System overview
- Architecture
- Core abstractions:
  - Alpha
  - Book
  - Allocator
- Example strategy description

DO NOT write README yet.

Only outline structure.

STOP.

---

# 9. ENGINEERING PRINCIPLES (MANDATORY)

- No look-ahead bias
- No full-sample statistical tests
- Reproducibility
- Modular design
- Separation of concerns

---

# 10. WHAT NOT TO DO

- Do NOT rewrite entire codebase
- Do NOT optimize performance
- Do NOT introduce new models
- Do NOT over-engineer

Focus ONLY on:

structure, clarity, and modularity

---

# 11. FINAL OBJECTIVE

Transform the project into:

> A modular cross-asset portfolio engine with an example statistical arbitrage strategy

Suitable for:

- GitHub showcase
- Quantitative research roles
- Portfolio engineering discussions

---

# END
