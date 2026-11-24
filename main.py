"""Single entry point for EAS 510 Project 2."""
import argparse
import random

import config


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


def parse_args():
    p = argparse.ArgumentParser(description="EAS 510 Project 2")
    p.add_argument("--data", choices=["sample", "full"], default="sample",
                   help="which dataset to run on (default: sample)")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(config.SEED)

    data_dir = config.SAMPLE_DATA_DIR if args.data == "sample" else config.FULL_DATA_DIR
    out_dir = config.RESULTS_DIR / args.data
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Data:    {data_dir}")
    print(f"Results: {out_dir}")
    # TODO: load data, train/evaluate model, write results


if __name__ == "__main__":
    main()
