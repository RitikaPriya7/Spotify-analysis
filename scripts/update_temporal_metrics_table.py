from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

DEFAULT_METRICS_PATH = Path("data/results/audio_temporal_classifier_metrics.csv")
DEFAULT_TEX_PATH = Path("report/aml_cw1_tables_and_figures.tex")

MODEL_DISPLAY = [
    ("Logistic Regression", "Logistic Regression"),
    ("Random Forest", "Random Forest"),
    ("Neural Networks", "Neural Network"),
    ("Support Vector Machines", "SVM (Approx RBF)"),
]

METRIC_ORDER = [
    ("accuracy", "Accuracy"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1", "F1 Score"),
    ("auc", "AUC"),
]

SPLITS = ["Train", "Test"]


def load_metrics(metrics_path: Path) -> Dict[Tuple[str, str], Dict[str, float]]:
    with metrics_path.open(newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        missing_cols = {"model", "split"} | {m for m, _ in METRIC_ORDER}
        if not missing_cols.issubset(reader.fieldnames or []):
            missing = missing_cols - set(reader.fieldnames or [])
            raise ValueError(
                f"Metrics file {metrics_path} is missing columns: {sorted(missing)}"
            )
        data: Dict[Tuple[str, str], Dict[str, float]] = {}
        for row in reader:
            split = row["split"]
            model = row["model"]
            if split not in SPLITS:
                continue
            key = (split, model)
            metrics_for_row: Dict[str, float] = {}
            for metric_key, _ in METRIC_ORDER:
                value = row.get(metric_key, "")
                try:
                    metrics_for_row[metric_key] = float(value)
                except (TypeError, ValueError):
                    metrics_for_row[metric_key] = math.nan
            data[key] = metrics_for_row
    return data


def format_value(value: float, precision: int) -> str:
    if math.isnan(value):
        return "--"
    return f"{value:.{precision}f}"


def build_table_rows(
    metrics_map: Dict[Tuple[str, str], Dict[str, float]], precision: int
) -> Tuple[str, List[str]]:
    rows: List[str] = []
    missing: List[str] = []

    for split_idx, split in enumerate(SPLITS):
        if split_idx > 0:
            rows.append("      \\midrule")

        for metric_key, metric_label in METRIC_ORDER:
            row_values: List[str] = []
            for display_name, model_key in MODEL_DISPLAY:
                lookup_key = (split, model_key)
                if lookup_key not in metrics_map:
                    missing.append(f"{split} / {model_key}")
                    row_values.append("--")
                    continue
                value = metrics_map[lookup_key][metric_key]
                row_values.append(format_value(value, precision))

            row_label = f"{split} {metric_label}"
            row_line = "      " + " & ".join([row_label, *row_values]) + r" \\"
            rows.append(row_line)

    return "\n".join(rows), missing


def build_table_block(rows: str) -> str:
    return (
        "\\newcommand{\\temporalMetricResultTable}{%\n"
        "  \\begin{table}[h]\n"
        "    \\centering\n"
        "    \\small\n"
        "    \\begin{tabular}{l|c|c|c|c}\n"
        "      \\toprule\n"
        "      Metric & Logistic Regression & Random Forest & Neural Networks & Support Vector Machines\\\\\n"
        "      \\midrule\n"
        f"{rows}\n"
        "      \\bottomrule\n"
        "    \\end{tabular}\n"
        "    \\caption{Comparison of model performance across all evaluation metrics using audio dataset split.}\n"
        "    \\label{tab:temporal-metric-results}\n"
        "  \\end{table}\n"
        "}\n"
    )


def replace_table_block(tex_path: Path, new_block: str) -> None:
    content = tex_path.read_text()
    pattern = re.compile(
        r"(\\newcommand{\s*\\temporalMetricResultTable}{%\s*\n)(.*?)(\n}\s*)",
        re.DOTALL,
    )
    match = pattern.search(content)
    if not match:
        raise ValueError(
            f"Could not locate temporalMetricResultTable definition in {tex_path}"
        )
    updated = content[: match.start()] + new_block.rstrip() + "\n" + content[match.end() :]
    tex_path.write_text(updated)


def main(metrics_path: Path, tex_path: Path, precision: int) -> None:
    metrics_map = load_metrics(metrics_path)
    rows, missing = build_table_rows(metrics_map, precision)
    if missing:
        raise ValueError(
            "Missing metric entries for: "
            + ", ".join(sorted(set(missing)))
        )
    new_block = build_table_block(rows)
    replace_table_block(tex_path, new_block)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Update the LaTeX audio metric table with Train/Test "
            "scores from the audio classifier notebook."
        )
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=DEFAULT_METRICS_PATH,
        help="Path to the CSV produced by the notebook metrics cell.",
    )
    parser.add_argument(
        "--tex-path",
        type=Path,
        default=DEFAULT_TEX_PATH,
        help="Path to aml_cw1_tables_and_figures.tex.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=3,
        help="Decimal precision for the table values.",
    )
    args = parser.parse_args()
    main(args.metrics_path, args.tex_path, args.precision)
