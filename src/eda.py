"""Week-1 EDA on the vectorized data: class balance, missing values, feature scales."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.features import GROUPS, feature_groups, feature_names


def run_eda(ds, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    X, y, month = ds.X, ds.y, ds.month

    # class balance, overall and by month
    by_month = pd.crosstab(pd.Series(month, name="month"),
                           pd.Series(np.where(y == 1, "malicious", "benign"), name="label"))
    by_month["malicious_frac"] = (by_month["malicious"] / by_month.sum(axis=1)).round(3)
    by_month.to_csv(out_dir / "class_balance_by_month.csv")

    # missing / non-finite values
    n_nan, n_inf = int(np.isnan(X).sum()), int(np.isinf(X).sum())

    # feature scales per column and per group
    names, groups = np.array(feature_names()), feature_groups()
    col = pd.DataFrame({"feature": names, "group": groups,
                        "min": X.min(0), "max": X.max(0), "mean": X.mean(0), "std": X.std(0),
                        "frac_zero": (X == 0).mean(0)})
    col.to_csv(out_dir / "feature_stats.csv", index=False)
    grp = col.groupby("group", sort=False).agg(
        n_features=("feature", "size"), min=("min", "min"), max=("max", "max"),
        median_std=("std", "median"), max_std=("std", "max"),
        mean_frac_zero=("frac_zero", "mean"), constant_cols=("std", lambda s: int((s == 0).sum())))
    grp = grp.reindex([g for g, _ in GROUPS])
    grp.to_csv(out_dir / "feature_scales_by_group.csv")

    # figures
    fig, ax = plt.subplots(figsize=(9, 3.5))
    by_month[["benign", "malicious"]].plot.bar(stacked=True, ax=ax, color=["#4C78A8", "#E45756"])
    ax.set_title("Subsample: samples per first-seen month"); ax.set_xlabel(""); ax.set_ylabel("rows")
    fig.tight_layout(); fig.savefig(out_dir / "class_balance_by_month.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.bar(col.index, np.log10(col["max"].abs() + 1), width=1.0)
    ax.set_title("Feature scale: log10(|max| + 1) per column"); ax.set_xlabel("feature index")
    start = 0
    for g, w in GROUPS:
        ax.axvline(start, color="grey", lw=0.5); ax.text(start + 2, ax.get_ylim()[1] * 0.92, g, fontsize=7, rotation=90)
        start += w
    fig.tight_layout(); fig.savefig(out_dir / "feature_scales.png", dpi=130); plt.close(fig)

    summary = {"rows": int(len(y)), "features": int(X.shape[1]),
               "malicious_frac": round(float(y.mean()), 4),
               "nan_values": n_nan, "inf_values": n_inf,
               "constant_columns": int((col["std"] == 0).sum()),
               "max_abs_value": float(np.abs(X).max())}
    pd.Series(summary).to_csv(out_dir / "summary.csv", header=["value"])
    return summary
