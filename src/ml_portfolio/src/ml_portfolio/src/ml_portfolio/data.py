"""Dataset generation and input validation."""
import numpy as np
from sklearn.datasets import make_classification

FEATURE_NAMES = [f"feature_{i}" for i in range(6)]


def make_dataset(n_samples=1200, random_state=42):
    """Create a deterministic binary classification dataset."""
    if n_samples < 100:
        raise ValueError("n_samples must be at least 100")

    X, y = make_classification(
        n_samples=n_samples,
        n_features=6,
        n_informative=4,
        n_redundant=1,
        n_repeated=0,
        n_classes=2,
        weights=[0.58, 0.42],
        class_sep=1.0,
        random_state=random_state,
    )
    return X.astype(float), y.astype(int)


def validate_features(X):
    """Validate a numeric feature matrix."""
    values = np.asarray(X, dtype=float)

    if values.ndim != 2:
        raise ValueError("Features must be a two-dimensional matrix")

    if values.shape[1] != len(FEATURE_NAMES):
        raise ValueError(
            f"Expected {len(FEATURE_NAMES)} features, got {values.shape[1]}"
        )

    if values.shape[0] == 0:
        raise ValueError("At least one row is required")

    if not np.isfinite(values).all():
        raise ValueError("Features contain NaN or infinite values")

    return values
