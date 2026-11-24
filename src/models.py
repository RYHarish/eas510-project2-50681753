"""Baseline models (week 2): L1/L2 logistic regression via SGD and a decision tree.

Logistic regression is fit with stochastic gradient descent on scaled features.
Scaling = signed log1p (tames heavy-tailed sizes/counts) then standardization.
Trees use raw features (they are invariant to monotone scaling).
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier

import config


def signed_log1p(X):
    return np.sign(X) * np.log1p(np.abs(X))


def linear_preprocessor():
    return [("log", FunctionTransformer(signed_log1p)), ("scale", StandardScaler())]


def baseline_models() -> dict[str, tuple[object, dict]]:
    """name -> (estimator, CV param grid)."""
    def logreg(penalty):
        return Pipeline(linear_preprocessor() + [(
            "clf", SGDClassifier(loss="log_loss", penalty=penalty, max_iter=config.LOGREG_MAX_ITER,
                                 tol=1e-4, average=True, random_state=config.SEED))])
    return {
        "logreg_l1": (logreg("l1"), {"clf__alpha": config.LOGREG_ALPHAS}),
        "logreg_l2": (logreg("l2"), {"clf__alpha": config.LOGREG_ALPHAS}),
        "decision_tree": (DecisionTreeClassifier(random_state=config.SEED), config.TREE_GRID),
    }
