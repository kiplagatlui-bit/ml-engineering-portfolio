import numpy as np
import pytest

from ml_portfolio.data import make_dataset, validate_features
from ml_portfolio.inference import predict
from ml_portfolio.modeling import train_and_evaluate


def test_dataset_is_reproducible():
    X1, y1 = make_dataset()
    X2, y2 = make_dataset()
    assert np.array_equal(X1, X2)
    assert np.array_equal(y1, y2)


def test_feature_validation_rejects_wrong_shape():
    with pytest.raises(ValueError):
        validate_features([[1, 2]])


def test_feature_validation_rejects_nan():
    with pytest.raises(ValueError):
        validate_features([[1, 2, 3, 4, 5, float("nan")]])


def test_training_persists_model_and_inference(tmp_path):
    metrics = train_and_evaluate(tmp_path)
    assert 0 <= metrics["accuracy"] <= 1
    assert (tmp_path / "model.joblib").exists()
    assert (tmp_path / "metrics.json").exists()

    result = predict(
        tmp_path / "model.joblib",
        [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]],
    )
    assert len(result) == 1
    assert result[0]["label"] in (0, 1)
    assert 0 <= result[0]["probability_class_1"] <= 1
