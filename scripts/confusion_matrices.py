#!/usr/bin/env python3
"""
Generate confusion matrices for each classifier on the Spotify dataset.
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ml.data import (
    AUDIO_FEATURES,
    RANK_COLUMN,
    create_classification_splits,
    load_classification_dataframe,
)
from ml.evaluation import build_confusion_matrix
from scripts.train_classifiers import (
    build_model_registry,
    parse_hidden_layers,
    validate_args,
)
from tqdm.auto import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute and visualize confusion matrices for four classifiers."
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
        default=Path("figures/confusion_matrices.png"),
        help="Where to save the confusion matrix grid.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    args = parse_args()
    validate_args(args)

    feature_cols = AUDIO_FEATURES
    model_df, target_col = load_classification_dataframe(
        args.data_path, feature_cols, RANK_COLUMN, args.top_k
    )
    train_df, dev_df, test_df = create_classification_splits(
        model_df,
        target_col,
        args.holdout_fraction,
        args.dev_share,
        args.random_state,
    )

    dataset_splits = {
        "Train": (train_df[feature_cols], train_df[target_col]),
        "Dev": (dev_df[feature_cols], dev_df[target_col]),
        "Test": (test_df[feature_cols], test_df[target_col]),
    }

    models = build_model_registry(args)

    confusion_records = []
    conf_matrices = {}
    model_items = list(models.items())
    train_bar = tqdm(
        total=len(model_items),
        desc="Training models for confusion matrices",
        leave=True,
    )
    eval_bar = tqdm(
        total=len(model_items),
        desc="Evaluating confusion matrices",
        leave=True,
    )
    for model_name, model in model_items:
        logging.info("Training %s for confusion matrix", model_name)
        train_bar.set_postfix_str(model_name)
        model.fit(dataset_splits["Train"][0], dataset_splits["Train"][1])
        train_bar.update(1)

        logging.info("Evaluating %s on Test split", model_name)
        eval_bar.set_postfix_str(model_name)
        cm = build_confusion_matrix(
            model, dataset_splits["Test"][0], dataset_splits["Test"][1]
        )
        eval_bar.update(1)

        conf_matrices[model_name] = cm
        confusion_records.append(
            {
                "model": model_name,
                "TN": cm[0, 0],
                "FP": cm[0, 1],
                "FN": cm[1, 0],
                "TP": cm[1, 1],
            }
        )

    train_bar.close()
    eval_bar.close()

    df = pd.DataFrame(confusion_records).set_index("model")
    print("Confusion matrices on Test split (counts):")
    print(df.to_string())

    n_models = len(conf_matrices)
    cols = 2
    rows = math.ceil(n_models / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(10, 5 * rows))
    axes = axes.flatten()

    for ax, (model_name, matrix) in zip(axes, conf_matrices.items()):
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["Pred 0", "Pred 1"],
            yticklabels=["Actual 0", "Actual 1"],
        )
        ax.set_title(model_name)

    for remaining_ax in axes[len(conf_matrices) :]:
        remaining_ax.axis("off")

    plt.tight_layout()
    args.figure_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.figure_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix figure to {args.figure_path}")


if __name__ == "__main__":
    main()
