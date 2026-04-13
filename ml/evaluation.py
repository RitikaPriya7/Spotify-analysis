from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_fscore_support, confusion_matrix

METRIC_COLUMNS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "auc",
    "top10_hit_rate",
]

def evaluate_pure_classifiers(model, X_train, y_train, X_test, y_test):
    results = {}

    # Train predictions
    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)[:,1] if hasattr(model, "predict_proba") else y_train_pred

    # Test predictions
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else y_test_pred

    # Train metrics
    results['Train Accuracy'] = accuracy_score(y_train, y_train_pred)
    results['Train Precision'] = precision_score(y_train, y_train_pred, zero_division=0)
    results['Train Recall'] = recall_score(y_train, y_train_pred, zero_division=0)
    results['Train F1 Score'] = f1_score(y_train, y_train_pred, zero_division=0)
    results['Train AUC'] = roc_auc_score(y_train, y_train_proba)

    # Test metrics
    results['Test Accuracy'] = accuracy_score(y_test, y_test_pred)
    results['Test Precision'] = precision_score(y_test, y_test_pred, zero_division=0)
    results['Test Recall'] = recall_score(y_test, y_test_pred, zero_division=0)
    results['Test F1 Score'] = f1_score(y_test, y_test_pred, zero_division=0)
    results['Test AUC'] = roc_auc_score(y_test, y_test_proba)

    return results



def evaluate_classifier(model, X, y) -> Dict[str, float]:
    y_pred = model.predict(X)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, y_pred, average="binary", zero_division=0
    )

    probas = None
    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(X)
        scores = probas[:, 1]
    elif hasattr(model, "decision_function"):
        decision = model.decision_function(X)
        scores = decision
        if decision.ndim == 1:
            probas = np.column_stack([-decision, decision])
        else:
            probas = decision
    else:
        scores = y_pred
        probas = np.column_stack([1 - y_pred, y_pred])
    try:
        auc = roc_auc_score(y, scores)
    except ValueError:
        auc = float("nan")

    positives = (y == 1).sum()
    hit_rate = ((y == 1) & (y_pred == 1)).sum() / positives if positives else float("nan")

    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "top10_hit_rate": hit_rate,
    }


def evaluate_classifier_splits(
    model,
    splits: Mapping[str, Tuple[Any, Any]],
) -> Dict[str, float]:
    """
    Evaluate ``model`` across multiple splits, prefixing metrics with the split name.
    """

    summary: Dict[str, float] = {}
    for split_name, (X_split, y_split) in splits.items():
        metrics = evaluate_classifier(model, X_split, y_split)
        for metric_name, value in metrics.items():
            summary[f"{split_name}_{metric_name}"] = value
    return summary


def build_confusion_matrix(model, X, y) -> np.ndarray:
    y_pred = model.predict(X)
    return confusion_matrix(y, y_pred)


def log_training_step(log_path: Path, record: Dict[str, Any]) -> None:
    """
    Append a single training/evaluation record to a CSV log.

    Parameters
    ----------
    log_path:
        Destination CSV file. Parent folders are created automatically.
    record:
        Dictionary containing metadata (model name, split, feature set, etc.)
        plus the metric values that should be persisted. All keys in `record`
        should remain consistent across writes to avoid header mismatches.
    """

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not log_path.exists()
    fieldnames = list(record.keys())

    with log_path.open("a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(record)
