"""
Small modelling helpers for the conversation → published analysis (notebook 03).
Thin wrappers around scikit-learn so the notebook stays readable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score


def evaluate_classifier(y_true, y_pred, y_prob=None) -> dict:
    """Print a classification report (+ ROC-AUC / PR-AUC if probabilities given)
    and return the headline metrics as a dict."""
    print(classification_report(y_true, y_pred, zero_division=0))
    out = {}
    if y_prob is not None:
        out["roc_auc"] = roc_auc_score(y_true, y_prob)
        out["pr_auc"] = average_precision_score(y_true, y_prob)
        print(f"ROC-AUC : {out['roc_auc']:.3f}")
        print(f"PR-AUC  : {out['pr_auc']:.3f}  (base rate = {np.mean(y_true):.3f})")
    return out


def odds_ratio_table(logreg, feature_names) -> pd.DataFrame:
    """Logistic-regression coefficients as odds ratios, sorted by effect size.
    odds_ratio > 1 ⇒ raises P(published); < 1 ⇒ lowers it. Assumes standardized
    numeric inputs, so ratios are per-1-SD (comparable across features)."""
    coef = np.ravel(logreg.coef_)
    tbl = pd.DataFrame({"feature": list(feature_names), "coef": coef})
    tbl["odds_ratio"] = np.exp(tbl["coef"])
    return tbl.sort_values("coef", ascending=False).reset_index(drop=True)
