import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA = PROJECT_ROOT / "ab_data.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
REPORT_DIR = OUTPUT_DIR / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
ALPHA = 0.05
CONFIDENCE_LEVEL = 0.95

# Bayesian Beta-Binomial priors (Jeffrey's non-informative: 0.5/0.5, uniform: 1/1)
BAYESIAN_PRIOR_ALPHA = 1
BAYESIAN_PRIOR_BETA = 1
BAYESIAN_SIMULATIONS = 50_000

# Resampling
BOOTSTRAP_ITERATIONS = 5_000
BOOTSTRAP_MAX_SAMPLE = 50_000   # cap per group (was 5 000 — too low)
PERMUTATION_ITERATIONS = 5_000

# ROPE: region of practical equivalence (absolute difference in conversion rate)
# ±0.002 = ±0.2 pp — meaningful for typical e-commerce / SaaS conversion rates
ROPE_LOWER = -0.002
ROPE_UPPER = 0.002

# Temporal binning
HOUR_BINS = [-1, 6, 12, 18, 24]
HOUR_LABELS = ["Night", "Morning", "Afternoon", "Evening"]

# Column role heuristics
CATEGORICAL_COLS = ["group", "landing_page"]
NUMERIC_COLS = ["converted"]
ID_COL = "user_id"
TIMESTAMP_COL = "timestamp"
TARGET_COL = "converted"
GROUP_COL = "group"
CONTROL_LABEL = "control"
TREATMENT_LABEL = "treatment"
