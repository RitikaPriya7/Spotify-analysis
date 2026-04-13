from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


METRICS_CSV = Path("all_classifier_metrics.csv")
TARGET_SPLIT = "holdout"  # use final evaluation split
METRIC_NAMES = ["accuracy", "precision", "recall", "f1", "auc"]
MODEL_ORDER = [
    "Logistic Regression",
    "Neural Network",
    "Random Forest",
    "SVM (Approx RBF)",
]

if not METRICS_CSV.exists():
    raise FileNotFoundError(f"Could not find {METRICS_CSV.resolve()}")

metrics_df = pd.read_csv(METRICS_CSV)
metrics_df["split"] = metrics_df["split"].str.strip()
metrics_df["feature_set"] = metrics_df["feature_set"].str.strip()

filtered = metrics_df[
    metrics_df["split"].str.lower() == TARGET_SPLIT.lower()
].copy()

if filtered.empty:
    raise ValueError(
        f"No rows found in {METRICS_CSV} for split '{TARGET_SPLIT}'. "
        "Update TARGET_SPLIT if a different label is required."
    )

records = []
for _, row in filtered.iterrows():
    dataset_name = row["feature_set"]
    model_name = row["model"]
    for metric_name in METRIC_NAMES:
        label = metric_name.upper() if metric_name == "auc" else metric_name.capitalize()
        records.append(
            {
                "Dataset": dataset_name,
                "Metric": label,
                "Model": model_name,
                "Value": row[metric_name],
            }
        )

plot_df = pd.DataFrame(records)

metric_order = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
datasets = sorted(plot_df["Dataset"].unique())
# Consistent blue-gray palette used previously; truncate/extend as needed
base_palette = ["#4f6d7a", "#75808a", "#9aa5b1"]
if len(datasets) <= len(base_palette):
    palette = base_palette[: len(datasets)]
else:
    extra = sns.color_palette("Greys", len(datasets) - len(base_palette))
    palette = base_palette + extra

output_dir = Path("report/figures/metric_panels")
output_dir.mkdir(parents=True, exist_ok=True)

for metric in metric_order:
    metric_df = plot_df[plot_df["Metric"] == metric]
    if metric_df.empty:
        print(f"Skipping {metric} plot; no data available.")
        continue
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(
        data=metric_df,
        x="Model",
        y="Value",
        hue="Dataset",
        hue_order=datasets,
        order=[m for m in MODEL_ORDER if m in metric_df["Model"].unique()],
        palette=palette,
        ax=ax,
        errorbar=None,
    )
    ax.set_title(f"{metric} ({TARGET_SPLIT.capitalize()} split)")
    ax.set_ylabel(metric)
    ax.set_xlabel("Model")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(title="Dataset", bbox_to_anchor=(1.02, 1), loc="upper left")
    else:
        legend = ax.get_legend()
        if legend:
            legend.remove()

    for container in ax.containers:
        ax.bar_label(container, fmt="{:.3f}", padding=2, fontsize=8)

    plt.tight_layout()
    output_path = output_dir / f"metric_{metric.lower()}.png"
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved plot to {output_path}")
