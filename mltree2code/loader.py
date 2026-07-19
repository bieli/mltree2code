from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import joblib

from mltree2code.exceptions import ModelLoadError, UnsupportedModelError

PathLike = Union[str, Path]

# Supported sklearn estimator class names (matched by type name).
_SUPPORTED_CLASSIFIERS = frozenset({"DecisionTreeClassifier"})
_SUPPORTED_REGRESSORS = frozenset({"DecisionTreeRegressor"})
_SUPPORTED = _SUPPORTED_CLASSIFIERS | _SUPPORTED_REGRESSORS


def load_model(path: PathLike) -> Any:
    path = Path(path)
    if not path.is_file():
        raise ModelLoadError(f"Model file not found: {path}")

    try:
        model = joblib.load(path)
    except Exception as exc:  # noqa: BLE001 - surface as ModelLoadError
        raise ModelLoadError(f"Failed to load model from {path}: {exc}") from exc

    validate_model(model)
    return model


def validate_model(model: Any) -> None:
    name = type(model).__name__
    if name not in _SUPPORTED:
        supported = ", ".join(sorted(_SUPPORTED))
        raise UnsupportedModelError(
            f"Unsupported model type '{name}'. Supported: {supported}. "
            "RandomForest / XGBoost / LightGBM support is planned."
        )

    if not hasattr(model, "tree_"):
        raise UnsupportedModelError(f"Model '{name}' has no tree_ attribute - is it fitted?")

    if getattr(model, "n_features_in_", None) is None:
        raise UnsupportedModelError(f"Model '{name}' does not look fitted.")


def is_classifier(model: Any) -> bool:
    return type(model).__name__ in _SUPPORTED_CLASSIFIERS
