"""Update \allMetricResults table in report/aml_cw1_tables_and_figures.tex.

Reads all_classifier_metrics.csv and regenerates the \allMetricResults
command so it lists cross-validation and holdout metrics for every
feature set and model.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

METRICS_CSV = Path("all_classifier_metrics.csv")
TARGET_FILE = Path("report/aml_cw1_tables_and_figures.tex")
COMMAND_NAME = "\\allMetricResults"

MODEL_ORDER = [
    "Logistic Regression",
    "Random Forest",
    "Neural Network",
    "SVM (Approx RBF)",
]

METRIC_ORDER = ["accuracy", "precision", "recall", "f1", "auc"]


class MetricsData:
    def __init__(self) -> None:
        self.data: Dict[str, Dict[str, Dict[str, Dict[str, str]]]] = {}

    def add_row(self, row: Dict[str, str]) -> None:
        feature = row["feature_set"].strip()
        split = row["split"].strip()
        model = row["model"].strip()
        self.data.setdefault(feature, {}).setdefault(split, {})[model] = row

    def get_value(self, feature: str, split: str, model: str, metric: str) -> str:
        try:
            val = float(self.data[feature][split][model][metric])
        except KeyError:
            return "0.000"
        return f"{val:.3f}"

    def feature_sets(self) -> List[str]:
        return sorted(self.data.keys())


def load_metrics() -> MetricsData:
    if not METRICS_CSV.exists():
        raise FileNotFoundError(f"Missing metrics CSV: {METRICS_CSV}")
    metrics = MetricsData()
    with METRICS_CSV.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metrics.add_row(row)
    return metrics


def build_table(metrics: MetricsData) -> str:
    lines: List[str] = ["\\newcommand{\\allMetricResults}{%"]
    lines.append("  \\begin{table}[h]")
    lines.append("    \\centering")
    lines.append("    \\small")
    lines.append("    \\begin{tabular}{l l|c|c|c|c}")
    lines.append("      \\toprule")
    lines.append(
        "      Feature set & Metric & Logistic Regression & Random Forest & Neural Network & SVM (Approx RBF)\\\\"
    )
    lines.append("      \\midrule")

    for feature in metrics.feature_sets():
        lines.append(f"      \\multicolumn{{6}}{{l}}{{\\textbf{{{feature}}}}}\\\\")
        for split_label in ["Cross", "Holdout"]:
            heading = "Cross-Validation" if split_label.lower() == "cross" else "Holdout"
            for metric in METRIC_ORDER:
                label = "F1 Score" if metric == "f1" else metric.title()
                metric_label = f"{heading} {label}"
                row_vals = [
                    metrics.get_value(feature, split_label, model, metric)
                    for model in MODEL_ORDER
                ]
                line = "      & {} & {} \\\\".format(metric_label, " & ".join(row_vals))
                lines.append(line)
        lines.append("      \\midrule")

    lines.pop()
    lines.append("      \\bottomrule")
    lines.append("    \\end{tabular}")
    lines.append("    \\caption{Cross-validation and holdout metrics across feature sets}")
    lines.append("    \\label{tab:all-split-metric-results}")
    lines.append("  \\end{table}")
    lines.append("}")
    return "\n".join(lines)


def replace_command(content: str, new_block: str) -> str:
    start_token = f"\\newcommand{{\\allMetricResults}}{{"
    start_idx = content.find(start_token)
    if start_idx == -1:
        raise ValueError("Could not find \\allMetricResults command in LaTeX file")
    brace_level = 0
    end_idx = start_idx
    for idx in range(start_idx, len(content)):
        char = content[idx]
        if char == '{':
            brace_level += 1
        elif char == '}':
            brace_level -= 1
            if brace_level == 0:
                end_idx = idx + 1
                break
    if brace_level != 0:
        raise ValueError("Unbalanced braces in LaTeX command definition")
    return content[:start_idx] + new_block + content[end_idx:]


def main() -> None:
    metrics = load_metrics()
    new_block = build_table(metrics)
    tex_content = TARGET_FILE.read_text()
    updated = replace_command(tex_content, new_block)
    TARGET_FILE.write_text(updated)
    print("Updated \\allMetricResults with fresh metrics.")


if __name__ == "__main__":
    main()
