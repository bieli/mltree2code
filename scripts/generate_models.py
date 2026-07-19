from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import (
    load_breast_cancer,
    load_diabetes,
    load_digits,
    load_iris,
    load_wine,
    make_classification,
    make_regression,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "models"


def _save(name: str, model) -> None:
    path = OUT / f"{name}.joblib"
    joblib.dump(model, path)
    print(f"  wrote {path.relative_to(ROOT)}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Generating models into {OUT}")

    iris = load_iris()
    for depth in (1, 2, 3, 5):
        clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
        clf.fit(iris.data, iris.target)
        _save(f"iris_depth{depth}", clf)

    wine = load_wine()
    for depth in (2, 5):
        clf = DecisionTreeClassifier(max_depth=depth, random_state=0)
        clf.fit(wine.data, wine.target)
        _save(f"wine_depth{depth}", clf)

    digits = load_digits()
    clf = DecisionTreeClassifier(max_depth=4, random_state=1)
    clf.fit(digits.data, digits.target)
    _save("digits_depth4", clf)

    bc = load_breast_cancer()
    clf = DecisionTreeClassifier(max_depth=5, random_state=7)
    clf.fit(bc.data, bc.target)
    _save("breast_cancer_depth5", clf)

    diabetes = load_diabetes()
    reg = DecisionTreeRegressor(max_depth=4, random_state=3)
    reg.fit(diabetes.data, diabetes.target)
    _save("diabetes_regressor", reg)

    X, y = make_classification(
        n_samples=200,
        n_features=8,
        n_informative=5,
        n_redundant=1,
        n_classes=2,
        random_state=11,
    )
    clf = DecisionTreeClassifier(max_depth=3, random_state=11)
    clf.fit(X, y)
    _save("synthetic_binary", clf)

    X, y = make_classification(
        n_samples=300,
        n_features=10,
        n_informative=6,
        n_classes=4,
        n_clusters_per_class=1,
        random_state=22,
    )
    clf = DecisionTreeClassifier(max_depth=4, random_state=22)
    clf.fit(X, y)
    _save("synthetic_multiclass", clf)

    X, y = make_regression(n_samples=250, n_features=6, noise=0.1, random_state=33)
    reg = DecisionTreeRegressor(max_depth=3, random_state=33)
    reg.fit(X, y)
    _save("synthetic_regressor", reg)

    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    clf = DecisionTreeClassifier(max_depth=1, random_state=0)
    clf.fit(X, y)
    _save("tiny_binary_depth1", clf)

    print("=== Done.")


if __name__ == "__main__":
    main()
