from __future__ import annotations

from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Training a full RBF SVM scales at least quadratically with the number of
# samples because it has to construct and store the entire kernel Gram matrix.
# For the Spotify dataset this becomes prohibitively slow. By default we now
# approximate the non-linear kernel with a Nyström feature map and train a
# linear SVM (SGDClassifier with hinge loss) in that transformed space. This
# keeps the decision boundary non-linear while reducing the complexity to
# roughly O(n_features * n_samples).


def build_svm(
    random_state: int = 42,
    c: float = 1.0,
    kernel: str = "rbf",
    gamma: str | float | None = "scale",
    approximate: bool = True,
    n_components: int = 600,
    sgd_alpha: float = 1e-4,
    sgd_max_iter: int = 2000,
) -> Pipeline:
    """
    Build an SVM-style classifier.

    Parameters
    ----------
    random_state:
        Seed for reproducibility across the scaler, Nyström projection and SGD.
    c:
        Regularization strength for the exact SVC path. Ignored when
        `approximate=True`.
    kernel:
        Kernel name for the exact SVC and for the Nyström feature map.
    gamma:
        Kernel coefficient. ``\"scale\"`` mirrors the default behaviour of
        scikit-learn's SVC. When using the approximation this value is passed to
        ``Nystroem`` if it is numeric, otherwise the default (`None`) is used.
    approximate:
        Whether to approximate the kernel with Nyström features (faster) or fall
        back to the exact ``SVC`` (slower but exact).
    n_components:
        Dimension of the Nyström feature map when `approximate=True`. Larger
        values improve fidelity but increase training cost.
    sgd_alpha / sgd_max_iter:
        Regularization strength and maximum iterations for the SGD classifier
        trained on top of the Nyström features.
    """

    steps = [("scaler", StandardScaler())]

    if approximate:
        approx_gamma = gamma if isinstance(gamma, (int, float)) else None
        steps.append(
            (
                "nystroem",
                Nystroem(
                    kernel=kernel,
                    gamma=approx_gamma,
                    n_components=n_components,
                    random_state=random_state,
                ),
            )
        )
        classifier = SGDClassifier(
            loss="hinge",
            alpha=sgd_alpha,
            max_iter=sgd_max_iter,
            tol=1e-3,
            class_weight="balanced",
            random_state=random_state,
        )
    else:
        classifier = SVC(
            C=c,
            kernel=kernel,
            gamma=gamma,
            class_weight="balanced",
            probability=True,
            random_state=random_state,
        )

    steps.append(("classifier", classifier))
    return Pipeline(steps=steps)
