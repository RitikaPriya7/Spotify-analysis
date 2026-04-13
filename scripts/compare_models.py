#!/usr/bin/env python3
"""
Train logistic regression and random forest baselines on the Spotify dataset
to predict whether a song reaches the Top-K, and compare their performance.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


AUDIO_FEATURES = [
    "Danceability",
    "Energy",
    "Loudness",
    "Speechiness",
    "Acousticness",
    "Instrumentalness",
    "Valence",
]
RANK_COLUMN = "Rank"
ID_COLUMN = "id"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train logistic regression and random forest classifiers on the "
            "Spotify dataset and compare their metrics."
        )
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/datasets/Spotify_Dataset_V3.csv"),
        help="Path to the full Spotify CSV file.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Songs ranked <= top_k are treated as positive examples.",
    )
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.4,
        help="Fraction of rows allocated to the combined dev/test holdout.",
    )
    parser.add_argument(
        "--dev-share",
        type=float,
        default=0.5,
        help="Fraction of the holdout assigned to the dev split (remainder -> test).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=51,
        help="Random seed used for shuffling and model initialization.",
    )
    parser.add_argument(
        "--rf-estimators",
        type=int,
        default=400,
        help="Number of trees in the random forest.",
    )
    parser.add_argument(
        "--rf-max-depth",
        type=int,
        default=None,
        help="Optional maximum tree depth for the random forest (None -> no limit).",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=Path("figures/compare_models_metrics.csv"),
        help="Optional path to write per-model metrics as CSV.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive.")
    if not 0 < args.holdout_fraction < 1:
        raise ValueError("--holdout-fraction must be between 0 and 1.")
    if not 0 < args.dev_share < 1:
        raise ValueError("--dev-share must be between 0 and 1.")
    if args.rf_estimators <= 0:
        raise ValueError("--rf-estimators must be positive.")
    if args.rf_max_depth is not None and args.rf_max_depth <= 0:
        raise ValueError("--rf-max-depth must be positive if specified.")


def load_model_dataframe(
    data_path: Path,
    feature_cols: Iterable[str],
    rank_col: str,
    top_k: int,
) -> Tuple[pd.DataFrame, str]:
    df = pd.read_csv(data_path, sep=";", encoding="utf-8-sig")
    df[rank_col] = pd.to_numeric(df[rank_col], errors="coerce")

    required_cols = list(feature_cols) + [rank_col, ID_COLUMN]
    model_df = (
        df.dropna(subset=required_cols)
        .drop_duplicates(subset=[ID_COLUMN])
        .copy()
    )
    target_col = f"is_top_{top_k}"
    model_df[target_col] = (model_df[rank_col] <= top_k).astype(int)
    model_df.reset_index(drop=True, inplace=True)
    return model_df, target_col


def create_splits(
    df: pd.DataFrame,
    target_col: str,
    holdout_fraction: float,
    dev_share: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, holdout_df = train_test_split(
        df,
        test_size=holdout_fraction,
        stratify=df[target_col],
        random_state=random_state,
    )
    dev_df, test_df = train_test_split(
        holdout_df,
        train_size=dev_share,
        stratify=holdout_df[target_col],
        random_state=random_state,
    )
    return train_df, dev_df, test_df


def build_models(
    random_state: int, rf_estimators: int, rf_max_depth: int | None
) -> Dict[str, Any]:
    logistic = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=500,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=random_state,
                ),
            ),
        ]
    )
    random_forest = RandomForestClassifier(
        n_estimators=rf_estimators,
        max_depth=rf_max_depth,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    return {
        "Logistic Regression": logistic,
        "Random Forest": random_forest,
    }


def evaluate_model(model, X, y) -> Dict[str, float]:
    y_pred = model.predict(X)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, y_pred, average="binary", zero_division=0
    )

    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)
    else:
        scores = y_pred
    try:
        auc = roc_auc_score(y, scores)
    except ValueError:
        auc = float("nan")

    positives = (y == 1).sum()
    top10_hits = ((y == 1) & (y_pred == 1)).sum()
    top10_hit_rate = top10_hits / positives if positives else float("nan")

    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "top10_hit_rate": top10_hit_rate,
    }


def main() -> None:
    configure_logging()
    args = parse_args()
    validate_args(args)

    feature_cols = AUDIO_FEATURES
    model_df, target_col = load_model_dataframe(
        args.data_path, feature_cols, RANK_COLUMN, args.top_k
    )
    positive_rate = model_df[target_col].mean()
    print(
        f"Loaded {len(model_df)} rows with target '{target_col}'. "
        f"Positive rate: {positive_rate:.2%}"
    )

    train_df, dev_df, test_df = create_splits(
        model_df, target_col, args.holdout_fraction, args.dev_share, args.random_state
    )
    print(
        f"Split sizes -> Train: {len(train_df)}, Dev: {len(dev_df)}, Test: {len(test_df)}"
    )

    models = build_models(args.random_state, args.rf_estimators, args.rf_max_depth)

    dataset_splits = {
        "Train": (train_df[feature_cols], train_df[target_col]),
        "Dev": (dev_df[feature_cols], dev_df[target_col]),
        "Test": (test_df[feature_cols], test_df[target_col]),
    }

    records = []
    for model_name, model in models.items():
        logging.info("Training %s", model_name)
        model.fit(dataset_splits["Train"][0], dataset_splits["Train"][1])
        logging.info("Finished training %s", model_name)
        for split_name, (X_split, y_split) in dataset_splits.items():
            logging.info("Evaluating %s on %s split", model_name, split_name)
            metrics = evaluate_model(model, X_split, y_split)
            metrics.update({"model": model_name, "split": split_name})
            records.append(metrics)

    metrics_df = pd.DataFrame(records)
    metrics_df = metrics_df.sort_values(["split", "model"])

    float_cols = ["accuracy", "precision", "recall", "f1", "auc", "top10_hit_rate"]
    formatters = {col: (lambda v, _c=col: f"{v:.3f}") for col in float_cols}

    print("\nPerformance comparison (higher is better):")
    print(metrics_df.to_string(index=False, formatters=formatters))

    if args.metrics_path:
        args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_df.to_csv(args.metrics_path, index=False)
        logging.info("Saved metrics to %s", args.metrics_path)


if __name__ == "__main__":
    main()
def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
