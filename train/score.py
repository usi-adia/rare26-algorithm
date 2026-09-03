"""Exact re-implementation of the RARE26 Grand Challenge metric (evaluation_Grand-Challenge.py):
median PPV@90%recall over n bootstrap draws, all negatives kept, positives resampled at 1:imbalance_ratio."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def ppv_at_recall(y_true, y_pred, recall=0.9) -> float:
    p, r, _ = precision_recall_curve(y_true, y_pred)
    return float(np.interp(recall, r[::-1], p[::-1]))


def bootstrap_metrics(y_true, y_pred, n_iterations=1000, imbalance_ratio=100, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred, dtype=np.float64)
    neg = np.where(y_true == 0)[0]; pos = np.where(y_true == 1)[0]
    n_pos = max(1, int(len(neg) / imbalance_ratio))
    rows = []
    for _ in range(n_iterations):
        idx = np.concatenate([neg, rng.choice(pos, size=n_pos, replace=True)])
        yt, yp = y_true[idx], y_pred[idx]
        rows.append((roc_auc_score(yt, yp), average_precision_score(yt, yp), ppv_at_recall(yt, yp)))
    rows = np.array(rows)
    return {
        "PPV@90RECALL": float(np.median(rows[:, 2])),
        "PPV@90RECALL_ci": [float(np.percentile(rows[:, 2], 2.5)), float(np.percentile(rows[:, 2], 97.5))],
        "AUROC": float(np.median(rows[:, 0])),
        "AUPRC": float(np.median(rows[:, 1])),
        "PPV@90RECALL_full": ppv_at_recall(y_true, y_pred),
        "AUROC_full": float(roc_auc_score(y_true, y_pred)),
        "n_neg": int(len(neg)), "n_pos": int(len(pos)), "n_pos_sampled": n_pos,
    }
