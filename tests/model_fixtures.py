"""Canonical joblib fixture names produced by scripts/generate_models.py.

Keep this list in sync with the generator — tests import it so parametrize
blocks cannot drift from what ``make models`` writes.
"""

from __future__ import annotations

# Order: shallow -> deep iris, then other datasets / synthetics.
MODEL_FIXTURES: tuple[str, ...] = (
    "iris_depth1",
    "iris_depth2",
    "iris_depth3",
    "iris_depth5",
    "wine_depth2",
    "wine_depth5",
    "digits_depth4",
    "breast_cancer_depth5",
    "diabetes_regressor",
    "synthetic_binary",
    "synthetic_multiclass",
    "synthetic_regressor",
    "tiny_binary_depth1",
)

# Stale names from older generator revisions (removed on regenerate).
OBSOLETE_FIXTURES: tuple[str, ...] = (
    "iris_classifiers_at_various_depths1",
    "iris_classifiers_at_various_depths2",
    "iris_classifiers_at_various_depths3",
    "iris_classifiers_at_various_depths5",
    "tiny_edge_cases_binary_depth1",
)
