"""Metrics: ROC-AUC, precision/recall/F1, confusion matrix, recall at fixed FPR."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)

import config


def recall_at_fpr(y, score, target_fpr):
    """Highest TPR achievable while keeping FPR <= target (read off the ROC curve)."""
    fpr, tpr, _ = roc_curve(y, score)
    ok = fpr <= target_fpr
    return float(tpr[ok].max()) if ok.any() else 0.0


def metrics(y, score, pred) -> dict:
    """`score` ranks samples (ROC-AUC, recall@FPR); `pred` is the model's default 0/1 decision."""
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out = {"n": len(y), "roc_auc": roc_auc_score(y, score),
           "precision": precision_score(y, pred, zero_division=0),
           "recall": recall_score(y, pred, zero_division=0),
           "f1": f1_score(y, pred, zero_division=0),
           "fpr": fp / max(fp + tn, 1), "tn": tn, "fp": fp, "fn": fn, "tp": tp}
    for t in config.FPR_TARGETS:
        out[f"recall@{t:.1%}fpr"] = recall_at_fpr(y, score, t)
    return {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else int(v)) for k, v in out.items()}


def per_month(y, score, pred, month) -> pd.DataFrame:
    rows = []
    for m in np.unique(month):
        k = month == m
        if len(np.unique(y[k])) < 2:
            continue
        rows.append({"month": m, **metrics(y[k], score[k], pred[k])})
    return pd.DataFrame(rows)
