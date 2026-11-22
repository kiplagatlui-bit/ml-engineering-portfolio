"""Batch inference using a persisted scikit-learn pipeline."""
from pathlib import Path

import joblib

from .data import validate_features


def predict(model_path: str | Path, values):
    """Return predicted class labels and positive-class probabilities."""
    model = joblib.load(model_path)
    X = validate_features(values)
    labels = model.predict(X).astype(int).tolist()
    probabilities = model.predict_proba(X)[:, 1].tolist()
    return [{"label": label, "probability_class_1": float(prob)} for label, prob in zip(labels, probabilities)]
