"""Random vs. time-based train/test splits over the same pool of samples."""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split

import config


def time_split(month: np.ndarray, train_end: str = config.TIME_TRAIN_END):
    """Train = first seen <= train_end; test = every later month."""
    train = np.flatnonzero(month <= train_end)
    test = np.flatnonzero(month > train_end)
    return train, test


def random_split(y: np.ndarray, n_train: int, seed: int = config.SEED):
    """Stratified random split with exactly n_train training rows."""
    idx = np.arange(len(y))
    train, test = train_test_split(idx, train_size=n_train, stratify=y, random_state=seed)
    return np.sort(train), np.sort(test)


def make_splits(y: np.ndarray, month: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    t_train, t_test = time_split(month)
    r_train, r_test = random_split(y, n_train=len(t_train))
    return {"random": (r_train, r_test), "time": (t_train, t_test)}


def describe_splits(splits, y, month):
    import pandas as pd
    rows = []
    for name, (tr, te) in splits.items():
        for part, idx in [("train", tr), ("test", te)]:
            rows.append({"split": name, "part": part, "n": len(idx),
                         "malicious_frac": round(float(y[idx].mean()), 4),
                         "first_month": min(month[idx]), "last_month": max(month[idx])})
    return pd.DataFrame(rows)
