"""Single entry point for EAS 510 Project 2: static-feature Windows malware detection.

Examples
    python main.py                       # everything on the bundled sample (~1 min)
    python main.py --data full           # full subsample (needs data/raw, see data/download.py)
    python main.py --data full --stage prepare   # only build the vectorized subsample + bundled sample
"""
import argparse
import random

import numpy as np

import config
from src.data import load_dataset, write_bundled_sample
from src.eda import run_eda
from src.experiment import run_experiment
from src.models import baseline_models
from src.splits import describe_splits, make_splits


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def parse_args():
    p = argparse.ArgumentParser(description="EAS 510 Project 2")
    p.add_argument("--data", choices=["sample", "full"], default="sample",
                   help="bundled sample (default) or the full EMBER subsample")
    p.add_argument("--stage", choices=["prepare", "eda", "baselines", "all"], default="all")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(config.SEED)
    out = config.RESULTS_DIR / args.data

    ds = load_dataset(args.data)
    print(f"[main] {args.data}: X={ds.X.shape}, malicious={ds.y.mean():.3f}, "
          f"months {min(ds.month)}..{max(ds.month)}")
    if args.data == "full" and args.stage in ("prepare", "all"):
        write_bundled_sample(ds)

    splits = make_splits(ds.y, ds.month)
    out.mkdir(parents=True, exist_ok=True)
    split_info = describe_splits(splits, ds.y, ds.month)
    split_info.to_csv(out / "splits.csv", index=False)
    print(split_info.to_string(index=False))
    if args.stage == "prepare":
        return

    if args.stage in ("eda", "all"):
        print("[main] EDA:", run_eda(ds, out / "eda"))
    if args.stage in ("baselines", "all"):
        res = run_experiment(ds, splits, baseline_models(), out / "baselines")
        cols = ["split", "model", "roc_auc", "precision", "recall", "f1",
                "recall@0.1%fpr", "recall@1.0%fpr", "cv_auc"]
        print(res[cols].to_string(index=False))


if __name__ == "__main__":
    main()
