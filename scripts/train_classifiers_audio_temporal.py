#!/usr/bin/env python3
"""
Train four baseline classifiers on the Spotify dataset and compare their metrics.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
from pathlib import Path
from typing import Dict, Sequence

import matplotlib.pyplot as plt
import pandas as pd
from tqdm.auto import tqdm

from ml.data import (
    AUDIO_FEATURES,
    TEMPORAL_FEATURES,
    RANK_COLUMN,
    create_classification_splits,
    load_classification_dataframe,
)
from ml.evaluation import evaluate_classifier
from ml.models import (
    build_logistic_regression,
    build_neural_network,
    build_random_forest,
    build_svm,
)


def parse_hidden_layers(arg: str) -> Sequence[int]:
    return tuple(int(part.strip()) for part in arg.split(",") if part.strip())


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train Logistic Regression, Neural Network, Random Forest, and "
            "SVM classifiers on the Spotify dataset and compare their metrics."
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
        default=42,
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
        "--nn-hidden",
        type=parse_hidden_layers,
        default=parse_hidden_layers("64,32"),
        help="Comma separated hidden-layer sizes for the neural network.",
    )
    parser.add_argument(
        "--nn-alpha",
        type=float,
        default=1e-3,
        help="L2 regularization term for the neural network.",
    )
    parser.add_argument(
        "--svm-c",
        type=float,
        default=1.0,
        help="Soft margin parameter C for the SVM.",
    )
    parser.add_argument(
        "--svm-kernel",
        type=str,
        default="rbf",
        help="Kernel to use for the SVM classifier.",
    )
    parser.add_argument(
        "--svm-gamma",
        type=str,
        default="scale",
        help="Gamma parameter for the SVM kernel ('scale', 'auto', or float).",
    )
    parser.add_argument(
        "--figure-path",
        type=Path,
        default=Path("figures/classifier_test_metrics.png"),
        help="Where to save the comparison figure.",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=Path("figures/classifier_metrics.csv"),
        help="Optional path to save per-model metrics as CSV.",
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
    if any(size <= 0 for size in args.nn_hidden):
        raise ValueError("--nn-hidden layers must be positive integers.")
    try:
        _ = float(args.svm_gamma) if args.svm_gamma not in {"scale", "auto"} else 0.0
    except ValueError as exc:
        raise ValueError("--svm-gamma must be 'scale', 'auto', or a float.") from exc


def build_model_registry(args: argparse.Namespace) -> Dict[str, object]:
    rf_max_depth = args.rf_max_depth if args.rf_max_depth is not None else None
    svm_gamma = (
        float(args.svm_gamma)
        if args.svm_gamma not in {"scale", "auto"}
        else args.svm_gamma
    )
    return {
        "Logistic Regression": build_logistic_regression(args.random_state),
        "Neural Network": build_neural_network(
            args.random_state, hidden_layer_sizes=args.nn_hidden, alpha=args.nn_alpha
        ),
        "Random Forest": build_random_forest(
            args.random_state,
            n_estimators=args.rf_estimators,
            max_depth=rf_max_depth,
        ),
        "SVM": build_svm(
            args.random_state,
            c=args.svm_c,
            kernel=args.svm_kernel,
            gamma=svm_gamma,
        ),
    }


def evaluate_models(models: Dict[str, object], dataset_splits):
    records = []
    model_items = list(models.items())
    train_bar = tqdm(
        total=len(model_items),
        desc="Training models",
        leave=True,
    )
    eval_bar = tqdm(
        total=len(model_items) * len(dataset_splits),
        desc="Evaluating splits",
        leave=True,
    )
    for model_name, model in model_items:
        logging.info("Starting training for %s", model_name)
        train_bar.set_postfix_str(model_name)
        X_train, y_train = dataset_splits["Train"]
        model.fit(X_train, y_train)
        logging.info("Finished training for %s", model_name)
        train_bar.update(1)

        for split_name, (X_split, y_split) in dataset_splits.items():
            logging.info("Evaluating %s on %s split", model_name, split_name)
            eval_bar.set_postfix_str(f"{model_name} | {split_name}")
            metrics = evaluate_classifier(model, X_split, y_split)
            metrics.update({"model": model_name, "split": split_name})
            records.append(metrics)
            eval_bar.update(1)

    train_bar.close()
    eval_bar.close()
    return pd.DataFrame(records)


def save_comparison_plot(
    metrics_df: pd.DataFrame,
    figure_path: Path,
) -> None:
    test_metrics = metrics_df[metrics_df["split"] == "Test"].set_index("model")
    columns_to_plot = ["accuracy", "precision", "recall", "f1"]

    plt.figure(figsize=(10, 6))
    test_metrics[columns_to_plot].plot(kind="bar")
    plt.title("Classifier performance on Test split")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(figure_path, dpi=300)
    plt.close()
    print(f"Saved comparison figure to {figure_path}")


def main() -> None:
    configure_logging()
    args = parse_args()
    validate_args(args)

    feature_cols = AUDIO_FEATURES + TEMPORAL_FEATURES
    model_df, target_col = load_classification_dataframe(
        args.data_path, feature_cols, RANK_COLUMN, args.top_k, FEATURE_SET_NAME="Temporal Features"
    )
    print(
        f"Loaded {len(model_df)} unique songs with Top-{args.top_k} label "
        f"({model_df[target_col].mean():.2%} positives)."
    )

    train_df, dev_df, test_df = create_classification_splits(
        model_df,
        target_col,
        args.holdout_fraction,
        args.dev_share,
        args.random_state,
    )
    print(
        f"Split sizes -> Train: {len(train_df)}, Dev: {len(dev_df)}, Test: {len(test_df)}"
    )

    dataset_splits = {
        "Train": (train_df[feature_cols], train_df[target_col]),
        "Dev": (dev_df[feature_cols], dev_df[target_col]),
        "Test": (test_df[feature_cols], test_df[target_col]),
    }

    models = build_model_registry(args)
    metrics_df = evaluate_models(models, dataset_splits)
    metrics_df = metrics_df.sort_values(["split", "model"]).reset_index(drop=True)

    formatters = {
        "accuracy": lambda v: f"{v:.3f}",
        "precision": lambda v: f"{v:.3f}",
        "recall": lambda v: f"{v:.3f}",
        "f1": lambda v: f"{v:.3f}",
        "auc": lambda v: f"{v:.3f}",
        "auc_ovr": lambda v: f"{v:.3f}",
        "top10_hit_rate": lambda v: f"{v:.3f}",
    }
    print("\nClassifier comparison (higher is better):")
    print(metrics_df.to_string(index=False, formatters=formatters))

    if args.metrics_path:
        args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_df.to_csv(args.metrics_path, index=False)
        logging.info("Saved metrics to %s", args.metrics_path)

    save_comparison_plot(metrics_df, args.figure_path)


if __name__ == "__main__":
    main()
