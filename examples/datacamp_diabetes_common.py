"""Shared DataCamp Pima diabetes decision-tree recipe.

Reproduces the tutorial setup from:
https://www.datacamp.com/tutorial/decision-tree-classification-python

The reference PNG ``examples/data/datacamp_diabetes_tree.png`` is the *unpruned*
``DecisionTreeClassifier()`` visualization from that article (wide/deep tree).
The article later also shows a pruned ``criterion="entropy", max_depth=3`` model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Column layout of the classic Pima Indians Diabetes CSV (no header).
COL_NAMES = (
    "pregnant",
    "glucose",
    "bp",
    "skin",
    "insulin",
    "bmi",
    "pedigree",
    "age",
    "label",
)

# Exact feature order from the DataCamp tutorial.
FEATURE_COLS = (
    "pregnant",
    "insulin",
    "bmi",
    "age",
    "glucose",
    "bp",
    "pedigree",
)

DATA_DIR = Path(__file__).resolve().parent / "data"
CSV_PATH = DATA_DIR / "pima-indians-diabetes.csv"
TREE_PNG = DATA_DIR / "datacamp_diabetes_tree.png"

# random_state=4 reproduces the article's published unpruned accuracy
# (0.6753246753246753) on current scikit-learn with this CSV + split.
UNPRUNED_RANDOM_STATE = 4
ARTICLE_UNPRUNED_ACCURACY = 0.6753246753246753
ARTICLE_PRUNED_ACCURACY = 0.7705627705627706


def load_pima_xy(csv_path: Path | None = None):
    """Load features/labels using the DataCamp column selection."""
    path = csv_path or CSV_PATH
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Expected vendored Pima Indians Diabetes CSV."
        )
    lines = path.read_text().splitlines()
    rows = [list(map(float, line.split(","))) for line in lines if line.strip()]
    data = np.asarray(rows, dtype=np.float64)
    idx = {name: i for i, name in enumerate(COL_NAMES)}
    X = data[:, [idx[c] for c in FEATURE_COLS]]
    y = data[:, idx["label"]].astype(int)
    return X, y


def train_test_split_pima(csv_path: Path | None = None):
    """70/30 split with ``random_state=1`` as in the article."""
    X, y = load_pima_xy(csv_path)
    return train_test_split(X, y, test_size=0.3, random_state=1)


def train_datacamp_diabetes_tree(
    *,
    pruned: bool = False,
    csv_path: Path | None = None,
) -> tuple[DecisionTreeClassifier, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Train the DataCamp diabetes tree (unpruned by default = reference image).

    Parameters
    ----------
    pruned:
        If False (default), fit ``DecisionTreeClassifier(random_state=4)`` —
        the large tree matching ``datacamp_diabetes_tree.png``.
        If True, fit ``DecisionTreeClassifier(criterion="entropy", max_depth=3)``
        as in the article's optimization section.
    """
    X_train, X_test, y_train, y_test = train_test_split_pima(csv_path)
    if pruned:
        clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)
    else:
        clf = DecisionTreeClassifier(random_state=UNPRUNED_RANDOM_STATE)
    clf.fit(X_train, y_train)
    return clf, X_train, X_test, y_train, y_test
