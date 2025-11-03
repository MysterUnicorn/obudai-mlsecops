import pytest

import os
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from ml_model import MLModel


@pytest.fixture(scope="module")
def wine_data():
    """Load and return the Wine Quality dataset."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    df = pd.read_csv(url, sep=";")
    return df


@pytest.fixture
def trained_model(wine_data):
    """Train a model and return it."""
    model = MLModel(random_seed=42)
    metrics = model.train(wine_data)
    return model, metrics


def test_train_returns_valid_metrics(trained_model):
    model, metrics = trained_model
    assert "in_sample_score" in metrics
    assert "out_of_sample_score" in metrics
    assert 0 <= metrics["in_sample_score"] <= 1
    assert 0 <= metrics["out_of_sample_score"] <= 1


def test_model_save_load_consistency(trained_model, wine_data):
    model_original, _ = trained_model
    X = wine_data.drop(columns=["quality"])
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.pkl")
        model_original.save_model(model_path)

        model_reloaded = MLModel()
        model_reloaded.load_model(model_path)

        pred_original = model_original.predict(X)
        pred_reloaded = model_reloaded.predict(X)

        np.testing.assert_array_equal(pred_original, pred_reloaded)


def test_training_reproducibility(wine_data):
    """Ensure same random seed produces identical results."""
    model1 = MLModel(random_seed=42)
    model2 = MLModel(random_seed=42)

    m1 = model1.train(wine_data)
    m2 = model2.train(wine_data)

    assert m1 == m2  # same seed -> same split -> same results


def test_save_model_creates_directories(trained_model):
    model, _ = trained_model
    with tempfile.TemporaryDirectory() as tmpdir:
        nested_path = Path(tmpdir) / "subdir" / "model.pkl"
        model.save_model(nested_path)
        assert nested_path.exists()


def test_predict_matches_train_f1_scores(wine_data):
    """
    Ensure predict() produces the same F1 scores as reported by train()
    on both training and test subsets.
    """
    seed = 42
    model = MLModel(random_seed=seed)

    # Train and capture reported metrics
    metrics = model.train(wine_data)

    # Recreate the same train/test split used internally
    # TODO: should be moved out to another function
    df = wine_data.drop_duplicates()
    df["target"] = np.where(df["quality"] < 7, 0, 1)
    y = df["target"]
    X = df.drop(["quality", "target"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, train_size=0.9, random_state=seed
    )

    # Predict on both splits using the trained model
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    f1_train = f1_score(y_train, y_pred_train, pos_label=1)
    f1_test = f1_score(y_test, y_pred_test, pos_label=1)

    # Compare with the F1 scores returned by train()
    assert np.isclose(f1_train, metrics["in_sample_score"], atol=1e-6), (
        f"Train F1 mismatch: {f1_train} vs {metrics['in_sample_score']}"
    )
    assert np.isclose(f1_test, metrics["out_of_sample_score"], atol=1e-6), (
        f"Test F1 mismatch: {f1_test} vs {metrics['out_of_sample_score']}"
    )
