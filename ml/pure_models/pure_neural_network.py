from __future__ import annotations

from typing import Any, Mapping, Sequence, Tuple

from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# HYPERPARAMETERS:
# - hidden_layer_sizes default to two layers with 64 and 32 neurons respectively, can be tuned based on validation performance
# - alpha set to 1e-3 to provide L2 regularization, helps prevent overfitting
# - max_iter set to 500 to allow sufficient training epochs for convergence


DEFAULT_NN_PARAMS: dict[str, Any] = {
    "random_state": 42,
    "hidden_layer_sizes": (64, 32),
    "alpha": 1e-3,
    "max_iter": 500,
}


def build_pure_neural_network(
    hyperparams: Mapping[str, Any] | None = None,
    **overrides: Any,
) -> Pipeline:
    final_params = DEFAULT_NN_PARAMS.copy()
    if hyperparams:
        final_params.update(hyperparams)
    if overrides:
        final_params.update(overrides)
    # typing helper: ensure tuples for layer sizes
    hidden_layers = final_params.get("hidden_layer_sizes")
    if isinstance(hidden_layers, Sequence) and not isinstance(hidden_layers, tuple):
        final_params["hidden_layer_sizes"] = tuple(hidden_layers)

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                MLPClassifier(**final_params),
            ),
        ]
    )
