from __future__ import annotations

from typing import Any, Mapping

from sklearn.ensemble import RandomForestClassifier

# HYPERPARAMETERS:
# - n_estimators set to 400 to provide a robust ensemble of trees
# - max_depth left as None to allow trees to expand fully, can be tuned based on validation performance
# - class_weight set to balanced to handle class imbalance by adjusting weights inversely proportional to class frequencies
# - n_jobs set to -1 to utilize all available CPU cores for parallel processing


DEFAULT_RF_PARAMS: dict[str, Any] = {
    "random_state": 42,
    "n_estimators": 400,
    "max_depth": None,
    "class_weight": "balanced",
    "n_jobs": -1,
}


def build_pure_random_forest(
    hyperparams: Mapping[str, Any] | None = None,
    **overrides: Any,
) -> RandomForestClassifier:
    final_params = DEFAULT_RF_PARAMS.copy()
    if hyperparams:
        final_params.update(hyperparams)
    if overrides:
        final_params.update(overrides)
    return RandomForestClassifier(**final_params)
