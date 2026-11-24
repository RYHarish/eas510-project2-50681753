"""Train/tune/evaluate models on the random and time splits and save everything."""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold

import config
from src.evaluate import metrics, per_month
from src.features import GROUPS, feature_groups


def tune_and_fit(est, grid, X, y):
    """Hyperparameters are chosen by CV on the training rows only; test rows are never seen."""
    cv = StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=config.SEED)
    gs = GridSearchCV(clone(est), grid, scoring=config.CV_SCORING, cv=cv,
                      n_jobs=config.N_JOBS, refit=True)
    gs.fit(X, y)
    return gs


def ranking_score(model, X):
    """Continuous score for ROC / recall@FPR. Logistic-regression probabilities saturate at
    exactly 1.0 for many samples (ties distort low-FPR metrics), so use the raw margin."""
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict_proba(X)[:, 1]


def linear_group_weights(model) -> dict:
    """Share of |coefficient| mass per feature group, plus L1 sparsity."""
    coef = np.abs(model.named_steps["clf"].coef_.ravel())
    groups = feature_groups()
    total = coef.sum() or 1.0
    out = {g: round(float(coef[groups == g].sum() / total), 4) for g, _ in GROUPS}
    out["nonzero_coefs"] = int((coef > 0).sum())
    return out


def run_experiment(ds, splits, models, out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, monthly, rocs, details = [], [], {}, {}
    for split_name, (tr, te) in splits.items():
        Xtr, ytr, Xte, yte = ds.X[tr], ds.y[tr], ds.X[te], ds.y[te]
        for model_name, (est, grid) in models.items():
            t0 = time.time()
            gs = tune_and_fit(est, grid, Xtr, ytr)
            score = ranking_score(gs.best_estimator_, Xte)
            pred = gs.best_estimator_.predict(Xte)
            elapsed = time.time() - t0
            m = metrics(yte, score, pred)
            rows.append({"split": split_name, "model": model_name, **m,
                         "cv_auc": round(gs.best_score_, 4), "train_n": len(tr),
                         "best_params": json.dumps(gs.best_params_), "seconds": round(elapsed, 1)})
            print(f"[exp] {split_name:6s} {model_name:14s} AUC={m['roc_auc']:.4f} "
                  f"R@1%={m['recall@1.0%fpr']:.3f} R@0.1%={m['recall@0.1%fpr']:.3f} "
                  f"({elapsed:.0f}s, {gs.best_params_})")
            fpr, tpr, _ = roc_curve(yte, score)
            rocs[(split_name, model_name)] = (fpr, tpr)
            pd.DataFrame(gs.cv_results_).drop(columns="params").to_csv(
                out_dir / f"cv_{split_name}_{model_name}.csv", index=False)
            if model_name.startswith("logreg"):
                details[f"{split_name}/{model_name}"] = linear_group_weights(gs.best_estimator_)
            if split_name == "time":
                pm = per_month(yte, score, pred, ds.month[te])
                pm.insert(0, "model", model_name)
                monthly.append(pm)

    res = pd.DataFrame(rows)
    res.to_csv(out_dir / "metrics.csv", index=False)
    if monthly:
        pd.concat(monthly).to_csv(out_dir / "time_split_by_month.csv", index=False)
    if details:
        pd.DataFrame(details).to_csv(out_dir / "logreg_coef_mass_by_group.csv")
    _plot_rocs(rocs, out_dir)
    if monthly:
        _plot_monthly(pd.concat(monthly), res, out_dir)
    return res


def _plot_rocs(rocs, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, split in zip(axes, ["random", "time"]):
        for (s, m), (fpr, tpr) in rocs.items():
            if s == split:
                ax.plot(fpr, tpr, label=m)
        ax.set_xscale("log"); ax.set_xlim(1e-4, 1)
        for t in config.FPR_TARGETS:
            ax.axvline(t, color="grey", ls=":", lw=0.8)
        ax.set_title(f"ROC ({split} split)"); ax.set_xlabel("false positive rate (log)")
        ax.legend(loc="lower right")
    axes[0].set_ylabel("true positive rate (recall)")
    fig.tight_layout(); fig.savefig(out_dir / "roc_curves.png", dpi=130); plt.close(fig)


def _plot_monthly(pm, res, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col in zip(axes, ["roc_auc", "recall@1.0%fpr"]):
        for i, (model, g) in enumerate(pm.groupby("model")):
            line, = ax.plot(g["month"], g[col], marker="o", label=f"{model} (time)")
            ref = res[(res.split == "random") & (res.model == model)][col]
            if len(ref):
                ax.axhline(float(ref.iloc[0]), color=line.get_color(), ls="--", lw=0.8)
        ax.set_title(f"{col} by test month (dashed = random split)")
        ax.tick_params(axis="x", rotation=45)
    axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_dir / "time_split_by_month.png", dpi=130); plt.close(fig)
