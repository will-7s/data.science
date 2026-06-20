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

BAYESIAN_PRIOR_ALPHA = 1
BAYESIAN_PRIOR_BETA = 1
BAYESIAN_SIMULATIONS = 50000

BOOTSTRAP_ITERATIONS = 5000
PERMUTATION_ITERATIONS = 5000

ROPE_LOWER = -0.001
ROPE_UPPER = 0.001

HOUR_BINS = [-1, 6, 12, 18, 24]
HOUR_LABELS = ["Night", "Morning", "Afternoon", "Evening"]

CATEGORICAL_COLS = ["group", "landing_page"]
NUMERIC_COLS = ["converted"]
ID_COL = "user_id"
TIMESTAMP_COL = "timestamp"
TARGET_COL = "converted"
GROUP_COL = "group"
CONTROL_LABEL = "control"
TREATMENT_LABEL = "treatment"
