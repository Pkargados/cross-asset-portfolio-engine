# COMPATIBILITY SHIM — do not add logic here.
# This file has been moved to regime/regime_detection.py (2026-03-30).
# All imports should use: from regime.regime_detection import ...
# This shim re-exports the public API for any legacy callers.
from regime.regime_detection import compute_regime_signals, compute_alpha_scale

__all__ = ["compute_regime_signals", "compute_alpha_scale"]
