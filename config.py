"""Seeds, settings, and hyperparameters."""
from pathlib import Path

SEED = 42

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
SAMPLE_DATA_DIR = DATA_DIR / "sample"
FULL_DATA_DIR = DATA_DIR / "full"
RESULTS_DIR = ROOT / "results"

# Hyperparameters
HPARAMS = {
    # "learning_rate": 1e-3,
    # "epochs": 10,
}
