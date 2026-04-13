from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Callable, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, clone
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline

from .evaluation import evaluate_classifier_splits
logger = logging.getLogger(__name__)
_SLUG_RE = re.compile(r"[^a-z0-9]+")


@dataclass
class EpochMetrics:
    """Container for the metrics we persist for every epoch."""

    epoch: int
    loss: float
    val_loss: float
    accuracy: float
    val_accuracy: float
    auc: float | None = None
    val_auc: float | None = None
    n_estimators: int | None = None


def _maybe_numpy(values: Iterable) -> np.ndarray:
    """Convert pandas/numpy objects to a numpy array without copying eagerly."""

    if hasattr(values, "to_numpy"):
        return values.to_numpy()
    return np.asarray(values)


def _probabilities_from_model(model: ClassifierMixin, X) -> np.ndarray | None:
    """
    Try to extract probability estimates for a dataset.

    Returns ``None`` if the model does not expose ``predict_proba`` nor
    ``decision_function``.
    """

    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    if hasattr(model, "decision_function"):
        decision = model.decision_function(X)
        decision = np.asarray(decision)
        if decision.ndim == 1:
            probs_pos = 1.0 / (1.0 + np.exp(-decision))
            return np.column_stack([1.0 - probs_pos, probs_pos])
        decision = decision - decision.max(axis=1, keepdims=True)
        exp_scores = np.exp(decision)
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)
    return None


def _score_vector(model: ClassifierMixin, X):
    """Best-effort retrieval of scores for AUC computation."""

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
    if hasattr(model, "decision_function"):
        decision = model.decision_function(X)
        decision = np.asarray(decision)
        if decision.ndim == 1:
            return decision
        if decision.ndim == 2 and decision.shape[1] >= 2:
            return decision[:, 1]
    if hasattr(model, "predict"):
        preds = model.predict(X)
        return np.asarray(preds)
    return None


def _evaluate_split(
    model: ClassifierMixin,
    X,
    y,
    *,
    sample_weight: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Compute loss/accuracy/auc for a split, falling back to NaNs."""

    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred, sample_weight=sample_weight)
    proba = _probabilities_from_model(model, X)
    loss = float("nan")
    if proba is not None:
        try:
            loss = log_loss(y, proba, sample_weight=sample_weight)
        except ValueError:
            loss = float("nan")
    auc = float("nan")
    scores = _score_vector(model, X)
    if scores is not None:
        try:
            auc = roc_auc_score(y, scores, sample_weight=sample_weight)
        except ValueError:
            auc = float("nan")
    return loss, accuracy, auc


def _fit_one_epoch(
    model: ClassifierMixin,
    X_train,
    y_train,
    *,
    classes: Sequence | None,
    partial_fit_initialized: bool,
    incremental_fit: Callable | None,
) -> bool:
    """
    Run one training epoch, using ``partial_fit`` when available.

    Returns a bool indicating whether the ``partial_fit`` path has already been
    initialised (i.e. had the ``classes`` argument supplied).
    """

    if incremental_fit is not None:
        kwargs = {}
        if not partial_fit_initialized and classes is not None:
            kwargs["classes"] = np.asarray(classes)
        incremental_fit(X_train, y_train, **kwargs)
        return True
    model.fit(X_train, y_train)
    return partial_fit_initialized


def _build_incremental_fitter(model: ClassifierMixin) -> Callable | None:
    """
    Return a callable that performs a single incremental update for ``model``.

    Supports estimators exposing ``partial_fit`` directly as well as ``Pipeline``
    objects whose transformers and final estimator all provide ``partial_fit``.
    """

    if hasattr(model, "partial_fit"):
        def _partial_fit(X, y, **kwargs):
            model.partial_fit(X, y, **kwargs)

        return _partial_fit

    if isinstance(model, Pipeline):
        transformers = [step for step_name, step in model.steps[:-1]]
        final_estimator = model.steps[-1][1]

        if not hasattr(final_estimator, "partial_fit"):
            return None
        for transformer in transformers:
            if not hasattr(transformer, "partial_fit") or not hasattr(
                transformer, "transform"
            ):
                return None

        def _pipeline_partial_fit(X, y, **kwargs):
            Xt = X
            for transformer in transformers:
                transformer.partial_fit(Xt)
                Xt = transformer.transform(Xt)
            final_estimator.partial_fit(Xt, y, **kwargs)

        return _pipeline_partial_fit

    return None


def train_model(
    model: ClassifierMixin,
    train_data,
    val_data,
    *,
    epochs: int = 10,
    classes: Sequence | None = None,
    history_path: str | Path | None = None,
    plot_path: str | Path | None = None,
    plot_title: str | None = None,
    early_stopping_patience: int | None = None,
    early_stopping_min_delta: float = 0.0,
    class_weights: dict | None = None,
) -> tuple[ClassifierMixin, pd.DataFrame]:
    """
    Train a classifier while recording metrics per epoch.

    Parameters
    ----------
    model:
        sklearn-style estimator that exposes ``fit``/``predict`` (and ideally
        ``partial_fit`` for incremental updates).
    train_data / val_data:
        Tuples of ``(X, y)`` for the training and validation splits.
    epochs:
        Number of passes through the data. Models without ``partial_fit`` will
        be re-fit from scratch each epoch.
    classes:
        Optional label set required by estimators that rely on
        ``partial_fit``. When omitted we derive it from ``y_train``.
    history_path:
        Optional CSV file where the epoch metrics should be persisted.
    plot_path / plot_title:
        Destination and optional title for the generated training curves.

    Returns
    -------
    model, history_df:
        The trained estimator and a ``DataFrame`` with one row per epoch.
    """

    X_train, y_train = train_data
    X_val, y_val = val_data

    if X_train is None or y_train is None:
        raise ValueError("Training data must be provided as (X, y).")
    if X_val is None or y_val is None:
        raise ValueError("Validation data must be provided as (X, y).")

    incremental_fit = _build_incremental_fitter(model)
    if classes is None and incremental_fit is not None:
        classes = np.unique(_maybe_numpy(y_train))

    history: list[EpochMetrics] = []
    partial_fit_initialized = False

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    epochs_run = 0

    train_sample_weights = _build_sample_weights(y_train, class_weights)

    for epoch in range(1, epochs + 1):
        partial_fit_initialized = _fit_one_epoch(
            model,
            X_train,
            y_train,
            classes=classes,
            partial_fit_initialized=partial_fit_initialized,
            incremental_fit=incremental_fit,
        )
        train_loss, train_accuracy, train_auc = _evaluate_split(
            model, X_train, y_train, sample_weight=train_sample_weights
        )
        val_loss, val_accuracy, val_auc = _evaluate_split(model, X_val, y_val)
        history.append(
            EpochMetrics(
                epoch=epoch,
                loss=train_loss,
                val_loss=val_loss,
                accuracy=train_accuracy,
                val_accuracy=val_accuracy,
                auc=train_auc,
                val_auc=val_auc,
            )
        )
        epochs_run += 1
        logger.info(
            (
                "Epoch %d/%d - loss: %.4f - val_loss: %.4f - "
                "accuracy: %.4f - val_accuracy: %.4f - "
                "auc: %.4f - val_auc: %.4f"
            ),
            epoch,
            epochs,
            train_loss,
            val_loss,
            train_accuracy,
            val_accuracy,
            train_auc,
            val_auc,
        )
        if (
            early_stopping_patience is not None
            and not np.isnan(val_loss)
        ):
            if val_loss + early_stopping_min_delta < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= early_stopping_patience:
                    logger.info(
                        "Early stopping triggered after %d epochs "
                        "(no val_loss improvement for %d epochs).",
                        epochs_run,
                        early_stopping_patience,
                    )
                    break

    history_df = pd.DataFrame(asdict(epoch) for epoch in history)

    if history_path is not None:
        history_path = Path(history_path)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_df.to_csv(history_path, index=False)
        logger.info("Saved training history to %s", history_path)

    if plot_path is not None:
        fig = plot_training_history(history_df, title=plot_title)
        plot_path = Path(plot_path)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved training plot to %s", plot_path)

    return model, history_df


def plot_training_history(
    history: pd.DataFrame | Sequence[EpochMetrics],
    *,
    title: str | None = None,
    figsize: tuple[int, int] = (15, 4),
):
    """
    Plot loss/accuracy curves for a training run.

    Parameters
    ----------
    history:
        Either the ``DataFrame`` returned by :func:`train_model` or a sequence
        of :class:`EpochMetrics`.
    title:
        Optional title displayed above the plots.
    figsize:
        Size of the matplotlib figure.

    Returns
    -------
    matplotlib.figure.Figure
    """

    if isinstance(history, pd.DataFrame):
        history_df = history.copy()
    else:
        history_df = pd.DataFrame(asdict(epoch) for epoch in history)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    axes[0].plot(history_df["epoch"], history_df["loss"], label="Train loss")
    axes[0].plot(history_df["epoch"], history_df["val_loss"], label="Val loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(history_df["epoch"], history_df["accuracy"], label="Train acc")
    axes[1].plot(history_df["epoch"], history_df["val_accuracy"], label="Val acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    if "auc" in history_df and "val_auc" in history_df:
        axes[2].plot(history_df["epoch"], history_df["auc"], label="Train AUC")
        axes[2].plot(history_df["epoch"], history_df["val_auc"], label="Val AUC")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("AUC")
        axes[2].legend()
    else:
        axes[2].axis("off")

    if title:
        fig.suptitle(title)

    fig.tight_layout()
    return fig


def _slugify_model_name(name: str) -> str:
    """Create a filesystem-friendly slug for ``name``."""

    slug = _SLUG_RE.sub("-", name.lower()).strip("-")
    return slug or "model"


def _balanced_class_weight_dict(
    y,
    *,
    classes: Sequence | None = None,
) -> dict:
    """Compute an explicit mapping equivalent to ``class_weight='balanced'``."""

    y_array = _maybe_numpy(y)
    if classes is None:
        classes = np.unique(y_array)
    classes = np.asarray(classes)
    counts = np.array([(y_array == cls).sum() for cls in classes], dtype=float)
    if np.any(counts == 0):
        raise ValueError("Cannot compute class weights because some classes are absent.")
    weights = len(y_array) / (len(classes) * counts)
    return dict(zip(classes, weights))


def _build_sample_weights(y, class_weights: dict | None) -> np.ndarray | None:
    """Construct per-sample weights from a class-weight mapping."""

    if not class_weights:
        return None
    y_array = _maybe_numpy(y)
    return np.asarray([class_weights.get(label, 1.0) for label in y_array], dtype=float)


def _maybe_apply_class_weights(
    model: ClassifierMixin,
    *,
    class_weights: dict | None,
) -> None:
    """
    Replace ``class_weight='balanced'`` with explicit weights for incremental fit.
    """

    if not class_weights:
        return

    targets: list[ClassifierMixin] = []
    if isinstance(model, Pipeline):
        targets.append(model.steps[-1][1])
    else:
        targets.append(model)

    for estimator in targets:
        if not hasattr(estimator, "get_params"):
            continue
        params = estimator.get_params()
        if params.get("class_weight") == "balanced":
            estimator.set_params(class_weight=class_weights)


def train_models(
    models: Mapping[str, ClassifierMixin],
    train_data,
    val_data,
    *,
    epochs: int = 10,
    classes: Sequence | None = None,
    history_dir: str | Path | None = None,
    plot_dir: str | Path | None = None,
    plot_title_template: str | None = None,
    early_stopping_patience: int | None = None,
    early_stopping_min_delta: float = 0.0,
    return_metrics: bool = False,
) -> tuple[dict[str, ClassifierMixin], dict[str, pd.DataFrame]] | tuple[
    dict[str, ClassifierMixin], dict[str, pd.DataFrame], dict[str, dict[str, float]]
]:
    """
    Train and log metrics for multiple estimators.

    Parameters
    ----------
    models:
        Mapping from model name to the estimator instance to train.
    train_data / val_data:
        ``(X, y)`` tuples for fitting and validation.
    epochs:
        Number of training epochs for each estimator.
    classes:
        Optional class labels for estimators that require ``partial_fit`` seeds.
    history_dir / plot_dir:
        Optional directories where per-model CSV histories and PNG plots will be
        stored. Directories are created automatically.
    plot_title_template:
        Optional template string used for the per-model plots. ``{name}``
        placeholders are replaced with the model's display name.

    Returns
    -------
    trained_models, histories:
        A dictionary with the trained estimators keyed by name and a second
        dictionary mapping names to their history ``DataFrame`` objects. When
        `return_metrics=True`, a third dictionary is returned containing the
        final train/dev metrics for each estimator.
    """

    trained_models: dict[str, ClassifierMixin] = {}
    histories: dict[str, pd.DataFrame] = {}
    final_metrics: dict[str, dict[str, float]] = {}

    history_dir_path = Path(history_dir) if history_dir is not None else None
    if history_dir_path is not None:
        history_dir_path.mkdir(parents=True, exist_ok=True)
    plot_dir_path = Path(plot_dir) if plot_dir is not None else None
    if plot_dir_path is not None:
        plot_dir_path.mkdir(parents=True, exist_ok=True)

    y_train = train_data[1]
    class_weights: dict | None = None
    try:
        class_weights = _balanced_class_weight_dict(y_train, classes=classes)
    except ValueError:
        class_weights = None

    for name, model in models.items():
        slug = _slugify_model_name(name)
        history_path = (
            history_dir_path / f"{slug}_history.csv"
            if history_dir_path is not None
            else None
        )
        plot_path = (
            plot_dir_path / f"{slug}_training.png"
            if plot_dir_path is not None
            else None
        )
        plot_title = (
            plot_title_template.format(name=name)
            if plot_title_template
            else f"{name} training history"
        )
        _maybe_apply_class_weights(model, class_weights=class_weights)
        logger.info("Training %s for %d epochs", name, epochs)
        trained_model, history_df = train_model(
            model,
            train_data,
            val_data,
            epochs=epochs,
            classes=classes,
            history_path=history_path,
            plot_path=plot_path,
            plot_title=plot_title,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=early_stopping_min_delta,
            class_weights=class_weights,
        )
        trained_models[name] = trained_model
        histories[name] = history_df
        if return_metrics:
            split_results = evaluate_classifier_splits(
                trained_model,
                {"train": train_data, "dev": val_data},
            )
            final_metrics[name] = split_results

    if return_metrics:
        return trained_models, histories, final_metrics
    return trained_models, histories


def train_random_forest_incremental(
    model: ClassifierMixin,
    train_data,
    val_data,
    *,
    epochs: int = 10,
    history_path: str | Path | None = None,
    plot_path: str | Path | None = None,
    plot_title: str | None = None,
    classes: Sequence | None = None,
) -> tuple[ClassifierMixin, pd.DataFrame]:
    """
    Incrementally build a warm-start random forest and log metrics per stage.

    The estimator must support ``warm_start`` and expose ``n_estimators``.
    """

    X_train, y_train = train_data
    X_val, y_val = val_data

    if not hasattr(model, "set_params") or not hasattr(model, "fit"):
        raise ValueError("Random forest model must expose set_params and fit.")

    target_estimators = getattr(model, "n_estimators", None)
    if not target_estimators or target_estimators <= 0:
        raise ValueError("Random forest must define a positive n_estimators value.")

    if hasattr(model, "estimators_") and getattr(model, "estimators_", None):
        model = clone(model)
    model.set_params(warm_start=True)

    tree_schedule = np.linspace(1, target_estimators, num=max(epochs, 1), dtype=int)
    tree_schedule = np.maximum.accumulate(tree_schedule)
    tree_schedule = [max(1, int(val)) for val in tree_schedule]

    deduped_schedule: list[int] = []
    for value in tree_schedule:
        if not deduped_schedule or value != deduped_schedule[-1]:
            deduped_schedule.append(value)
    tree_schedule = deduped_schedule
    total_stages = len(tree_schedule)

    history: list[EpochMetrics] = []
    class_weights: dict | None = None
    try:
        class_weights = _balanced_class_weight_dict(y_train, classes=classes)
    except ValueError:
        class_weights = None
    train_sample_weights = _build_sample_weights(y_train, class_weights)
    for idx, n_estimators in enumerate(tree_schedule, start=1):
        model.set_params(n_estimators=n_estimators)
        model.fit(X_train, y_train)
        train_loss, train_accuracy, train_auc = _evaluate_split(
            model, X_train, y_train, sample_weight=train_sample_weights
        )
        val_loss, val_accuracy, val_auc = _evaluate_split(model, X_val, y_val)
        history.append(
            EpochMetrics(
                epoch=idx,
                loss=train_loss,
                val_loss=val_loss,
                accuracy=train_accuracy,
                val_accuracy=val_accuracy,
                n_estimators=n_estimators,
                auc=train_auc,
                val_auc=val_auc,
            )
        )
        logger.info(
            (
                "Forest stage %d/%d (trees=%d) - "
                "loss: %.4f - val_loss: %.4f - accuracy: %.4f - "
                "val_accuracy: %.4f - auc: %.4f - val_auc: %.4f"
            ),
            idx,
            total_stages,
            n_estimators,
            train_loss,
            val_loss,
            train_accuracy,
            val_accuracy,
            train_auc,
            val_auc,
        )

    history_df = pd.DataFrame(asdict(epoch) for epoch in history)

    if history_path is not None:
        history_path = Path(history_path)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_df.to_csv(history_path, index=False)
        logger.info("Saved random forest history to %s", history_path)

    if plot_path is not None:
        fig = plot_training_history(history_df, title=plot_title)
        plot_path = Path(plot_path)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved random forest training plot to %s", plot_path)

    return model, history_df
