from __future__ import annotations

from typing import Any, Mapping

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# OUR HYPERPARAMETERS ARE:
# - max_iter, run over 500 epochs
# - class_weight set to balanced to account for any class imbalance, it does this by adjusting weights inversely proportional to class frequencies
# - solver set to liblinear, good for small datasets and binary classification
# - reguralization and other hyperparameters are left as default


# IF hyperparameters dictionary passed, use that otherwise use default values
DEFAULT_LOGREG_PARAMS: dict[str, Any] = {
    "random_state": 42,
    "class_weight": "balanced",
    "max_iter": 500,
    "solver": "liblinear",
}


def build_pure_logistic_regression(
    hyperparams: Mapping[str, Any] | None = None,
    **overrides: Any,
) -> Pipeline:
    final_params = DEFAULT_LOGREG_PARAMS.copy()
    if hyperparams:
        final_params.update(hyperparams)
    if overrides:
        final_params.update(overrides)

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(**final_params),
            ),
        ]
    )
