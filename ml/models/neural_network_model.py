from __future__ import annotations

from typing import Sequence, Tuple

from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# BEST CONFIG FOR NN:
# Best Result with most dramatic increase in val_auc:
# hidden_layers            (32,)
# alpha                  0.0001
# activation                tanh
# learning_rate         adaptive
# learning_rate_init        0.01
# momentum                   0.9
# dropout_rate               0.2
# best_val_loss         0.191217
# dev_accuracy          0.951863
# dev_auc               0.571685
# epochs_run                   4


def build_neural_network(
    random_state: int,
    hidden_layer_sizes: Sequence[int] | Tuple[int, ...] = (32,),
    alpha: float = 1e-4,
    activation: str = "tanh",
    learning_rate: str = "adaptive",
    learning_rate_init: float = 1e-2,
    momentum: float = 0.9,
    solver: str = "sgd",
    max_iter: int = 500,
) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                MLPClassifier(
                    hidden_layer_sizes=tuple(hidden_layer_sizes),
                    alpha=alpha,
                    activation=activation,
                    learning_rate=learning_rate,
                    learning_rate_init=learning_rate_init,
                    momentum=momentum,
                    solver=solver,
                    max_iter=max_iter,
                    # hidden_layer_sizes= (64, 32),
                    # alpha=1e-3,
                    # max_iter=500,
                    random_state=random_state,
                ),
            ),
        ]
    )
