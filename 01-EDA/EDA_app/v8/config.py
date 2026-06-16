"""
config.py  —  Centralised thresholds and configuration constants.

All tunable parameters live here so they can be changed in one place
without hunting through the codebase.
"""
from __future__ import annotations

# ── Sampling threshold ────────────────────────────────────────────────────
SAMPLE_THRESHOLD: int = 5_000           # rows; beyond this, heavy
                                         # computations automatically sample
# ── Sampling ─────────────────────────────────────────────────────────────
SAMPLE_N:           int = 5_000         # rows for random subsample
SAMPLE_SEED:        int = 42            # global RNG seed for sampling

SUBSAMPLE_THRESHOLD: int = 5_000        # normality test switches to
                                        # stratified subsampling beyond this
SUBSAMPLE_N:        int = 2_000         # size of each stratified subsample
SUBSAMPLE_REPS:     int = 5             # number of stratified subsamples
SUBSAMPLE_BINS:     int = 20            # quantile strata for draw
SUBSAMPLE_SEED:     int = 42            # RNG seed for subsampling

# ── Monte Carlo ──────────────────────────────────────────────────────────
MC_REPS: int = 100                       # Lilliefors null-distribution reps
MC_SEED: int = 0                         # Lilliefors RNG seed

# ── Chart limits ─────────────────────────────────────────────────────────
SCATTER_MAX: int = 10_000               # scatter points; beyond this
SCATTER_MAX_LIGHT: int = 3_000          # scatter points in lightweight mode
BAR_MAX_CATS: int = 50                  # max categories in bar chart;
                                        # remainder grouped as "others"
PIE_MAX_CATS: int = 20                  # max categories in pie chart

# ── Statistics ───────────────────────────────────────────────────────────
ALPHA: float = 0.05                     # significance level for tests

# ── Misc ─────────────────────────────────────────────────────────────────
LIGHTWEIGHT_SLOW_COUNT: int = 2         # consecutive slow computations
                                        # before lightweight mode activates
DEFAULT_LOG_PATH: str = "time_budget_log.json"  # unused (legacy)
