from __future__ import annotations

from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_logistic_regression(
    random_state: int,
    alpha: float = 0.01,
    learning_rate: str = "constant",
    eta0: float = 0.01,
    penalty: str = "elasticnet",
    l1_ratio: float = 0.3,
    power_t: float = 0.5,
    max_iter: int = 10,
) -> Pipeline:
    """
    Build an SGD-based logistic regression classifier.

    Using SGDClassifier with ``log_loss`` allows us to leverage ``partial_fit``
    so the notebook training loop can report per-epoch metrics.
    """

    return Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            (
                "classifier",
                SGDClassifier(
                    loss="log_loss",
                    penalty=penalty,
                    alpha=alpha,
                    learning_rate=learning_rate,
                    eta0=eta0,
                    l1_ratio=l1_ratio,
                    power_t=power_t,
                    tol=None,
                    warm_start=True,
                    # class_weight="balanced",
                    random_state=random_state,
                    max_iter=500
                ),
            ),
        ]
    )
