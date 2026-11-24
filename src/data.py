"""Data loading: index the raw EMBER files, draw the subsample, vectorize, and cache.

Full-data flow (needs data/raw/ember2018/, see data/download.py):
    build_index()      one fast regex pass over all 1M rows -> (file, line, month, label)
    draw_subsample()   stratified (month x label) random draw of SUBSAMPLE_SIZE rows
    build_processed()  parse + vectorize only the drawn rows -> data/processed/
Sample flow: load_dataset("sample") reads the bundled data/sample/ember_sample.npz.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import config
from src.features import N_FEATURES, vectorize

_META_RE = re.compile(rb'"sha256":\s*"([0-9a-f]+)".*?"appeared":\s*"([^"]*)",\s*"label":\s*(-?\d)')


@dataclass
class Dataset:
    X: np.ndarray          # (n, 2381) float32
    y: np.ndarray          # (n,) int8, 0 = benign, 1 = malicious
    month: np.ndarray      # (n,) str "YYYY-MM" first-seen month
    sha256: np.ndarray     # (n,) str


# ---------------------------------------------------------------------------- index
def build_index(raw_dir: Path = config.RAW_DIR) -> pd.DataFrame:
    """One row per raw record with its file, line number, month and label."""
    files = sorted(raw_dir.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No EMBER .jsonl files in {raw_dir}. Run `python data/download.py`.")
    rows = []
    for f in files:
        with open(f, "rb") as fh:
            for i, line in enumerate(fh):
                m = _META_RE.search(line, 0, 600)
                rows.append((f.name, i, m.group(1).decode(), m.group(2).decode(), int(m.group(3))))
    return pd.DataFrame(rows, columns=["file", "line", "sha256", "month", "label"])


def draw_subsample(index: pd.DataFrame, n: int = config.SUBSAMPLE_SIZE,
                   seed: int = config.SEED) -> pd.DataFrame:
    """Stratified random draw: every (month, label) cell keeps the same fraction."""
    pool = index[(index.label >= 0) & (index.month >= config.POOL_START_MONTH)]
    frac = n / len(pool)
    sub = pool.groupby(["month", "label"]).sample(frac=frac, random_state=seed)
    return sub.sort_values(["file", "line"]).reset_index(drop=True)


def _read_lines(path: Path, wanted: set[int]):
    with open(path) as fh:
        for i, line in enumerate(fh):
            if i in wanted:
                yield json.loads(line)


def build_processed(out_dir: Path = config.PROCESSED_DIR, batch: int = 5000) -> Dataset:
    """Index -> subsample -> vectorize. Writes X.npy, meta.csv, index summary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    print("[data] indexing raw files ...")
    index = build_index()
    index.groupby(["month", "label"]).size().rename("n").to_csv(out_dir / "population_counts.csv")

    sub = draw_subsample(index)
    n_pool = int(((index.label >= 0) & (index.month >= config.POOL_START_MONTH)).sum())
    print(f"[data] subsample: {len(sub):,} of {n_pool:,} labeled 2018 rows")

    X = np.lib.format.open_memmap(out_dir / "X.npy", mode="w+", dtype=np.float32,
                                  shape=(len(sub), N_FEATURES))
    pos = 0
    for fname, grp in sub.groupby("file", sort=True):
        recs = []
        for rec in _read_lines(config.RAW_DIR / fname, set(grp.line)):
            recs.append(rec)
            if len(recs) == batch:
                X[pos:pos + batch] = vectorize(recs)
                pos += batch
                recs = []
        if recs:
            X[pos:pos + len(recs)] = vectorize(recs)
            pos += len(recs)
        print(f"[data]   {fname}: {len(grp):,} rows vectorized ({pos:,} total)")
    X.flush()
    assert pos == len(sub)
    sub.to_csv(out_dir / "meta.csv", index=False)  # documents exactly which rows were drawn
    return load_processed(out_dir)


def load_processed(out_dir: Path = config.PROCESSED_DIR) -> Dataset:
    meta = pd.read_csv(out_dir / "meta.csv", dtype={"month": str})
    X = np.load(out_dir / "X.npy", mmap_mode="r")
    return Dataset(np.asarray(X), meta.label.to_numpy(np.int8),
                   meta.month.to_numpy(str), meta.sha256.to_numpy(str))


# ---------------------------------------------------------------------------- sample
def write_bundled_sample(ds: Dataset, n: int = config.BUNDLED_SAMPLE_SIZE,
                         path: Path = config.SAMPLE_FILE, seed: int = config.SEED) -> None:
    """Small stratified (month x label) slice of the subsample, committed to the repo."""
    df = pd.DataFrame({"month": ds.month, "label": ds.y})
    frac = n / len(df)
    idx = df.groupby(["month", "label"]).sample(frac=frac, random_state=seed).index.sort_values()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X=ds.X[idx], y=ds.y[idx], month=ds.month[idx], sha256=ds.sha256[idx])
    print(f"[data] wrote bundled sample: {len(idx):,} rows -> {path}")


def load_dataset(which: str) -> Dataset:
    if which == "sample":
        z = np.load(config.SAMPLE_FILE, allow_pickle=False)
        return Dataset(z["X"], z["y"].astype(np.int8), z["month"], z["sha256"])
    if (config.PROCESSED_DIR / "X.npy").exists() and (config.PROCESSED_DIR / "meta.csv").exists():
        return load_processed()
    return build_processed()
