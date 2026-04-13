from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier

# BEST CONFIG (from rf_hyperparameter_search_results.csv):
# - n_estimators: 600
# - max_depth: 20
# - min_samples_leaf: 1
# - max_features: sqrt
# - bootstrap: True
# - class_weight stays balanced to offset label imbalance.
# - n_jobs set to -1 to utilize all cores.


def build_random_forest(
    random_state: int,
    n_estimators: int = 600,
    max_depth: int | None = 20,
    min_samples_leaf: int = 1,
    max_features: str | float | None = "sqrt",
    bootstrap: bool = True,
    warm_start: bool = True,
) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=bootstrap,
        class_weight="balanced",
        n_jobs=-1,
        warm_start=warm_start,
        # n_estimators=400,
        # max_depth=None,
        # class_weight="balanced",
        # n_jobs=-1,
        random_state=random_state,
    )
