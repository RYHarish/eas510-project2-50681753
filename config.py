"""Seeds, settings, and hyperparameters (single source of truth for every run)."""
from pathlib import Path

SEED = 42

# ---- paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "ember2018"           # extracted EMBER 2018 v2 JSONL (gitignored)
PROCESSED_DIR = DATA_DIR / "processed"             # vectorized subsample (gitignored)
SAMPLE_FILE = DATA_DIR / "sample" / "ember_sample.npz"  # bundled, committed
RESULTS_DIR = ROOT / "results"

EMBER_URL = "https://ember.elastic.co/ember_dataset_2018_2.tar.bz2"

# ---- subsampling ---------------------------------------------------------------
# Pool = labeled samples (label 0/1) first seen in 2018. The 50k pre-2018 files are
# benign-only, so they are excluded to keep both splits drawn from the same period.
POOL_START_MONTH = "2018-01"
SUBSAMPLE_SIZE = 150_000      # rows drawn from the ~750k-row pool, stratified by month x label
BUNDLED_SAMPLE_SIZE = 2_000   # rows in data/sample/ember_sample.npz (drawn from the subsample)

# ---- splits --------------------------------------------------------------------
# Time split: train on samples first seen up to and including TIME_TRAIN_END,
# test on every later month (one curve point per month).
TIME_TRAIN_END = "2018-06"
# Random split: stratified random split of the same pool with the SAME number of
# training rows as the time split, so the only difference is how rows are assigned.

# ---- evaluation ----------------------------------------------------------------
FPR_TARGETS = (0.001, 0.01)   # report recall at 0.1% and 1% false-positive rate
CV_FOLDS = 3
CV_SCORING = "roc_auc"
N_JOBS = 6   # parallel CV fits; each logreg worker holds a float64 copy of the training data

# ---- week-2 baselines: hyperparameter grids (tuned by CV on the training period only)
LOGREG_ALPHAS = [1e-5, 1e-4, 1e-3, 1e-2]          # SGD regularization strength
LOGREG_MAX_ITER = 100
TREE_GRID = {"max_depth": [10, 20, 40], "min_samples_leaf": [1, 10, 50]}
