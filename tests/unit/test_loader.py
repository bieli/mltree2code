"""Unit tests for model loader."""

from __future__ import annotations

from pathlib import Path

import joblib
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from mltree2code.exceptions import ModelLoadError, UnsupportedModelError
from mltree2code.loader import load_model, validate_model


def test_load_joblib(tmp_path: Path, iris_model):
    model, _ = iris_model
    path = tmp_path / "tree.joblib"
    joblib.dump(model, path)
    loaded = load_model(path)
    assert type(loaded).__name__ == "DecisionTreeClassifier"


def test_missing_file():
    with pytest.raises(ModelLoadError):
        load_model("/nonexistent/path/model.joblib")


def test_unsupported_random_forest(iris_model):
    model, iris = iris_model
    rf = RandomForestClassifier(n_estimators=3, random_state=0)
    rf.fit(iris.data, iris.target)
    with pytest.raises(UnsupportedModelError):
        validate_model(rf)


def test_unfitted_tree():
    clf = DecisionTreeClassifier()
    with pytest.raises(UnsupportedModelError):
        validate_model(clf)
