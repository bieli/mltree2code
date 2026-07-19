from __future__ import annotations

from pathlib import Path

import joblib
import pytest
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

MODELS_DIR = Path(__file__).parent / "models"


@pytest.fixture
def iris_model():
    iris = load_iris()
    clf = DecisionTreeClassifier(max_depth=2, random_state=42)
    clf.fit(iris.data, iris.target)
    return clf, iris


@pytest.fixture
def regressor_model():
    import numpy as np

    X = np.linspace(0, 10, 50).reshape(-1, 1)
    y = (X[:, 0] > 5).astype(float) * 3.0 + 1.0
    reg = DecisionTreeRegressor(max_depth=2, random_state=0)
    reg.fit(X, y)
    return reg, X, y


@pytest.fixture
def models_dir() -> Path:
    return MODELS_DIR


def load_fixture(name: str):
    path = MODELS_DIR / f"{name}.joblib"
    if not path.exists():
        pytest.skip(f"Fixture missing: {path} — run scripts/generate_models.py")
    return joblib.load(path)
