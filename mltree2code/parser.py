from __future__ import annotations

from typing import Any

import numpy as np

from mltree2code.exceptions import ParseError
from mltree2code.ir import Leaf, Node, TreeIR, TreeNode
from mltree2code.loader import is_classifier


def parse_sklearn_tree(
    model: Any,
    *,
    feature_names: list[str] | None = None,
    class_names: list[str] | None = None,
) -> TreeIR:
    if not hasattr(model, "tree_"):
        raise ParseError("Model has no tree_ attribute")

    tree = model.tree_
    classifier = is_classifier(model)

    # Resolve feature names
    if feature_names is None and hasattr(model, "feature_names_in_"):
        feature_names = [str(n) for n in model.feature_names_in_]

    # Resolve class names
    if classifier and class_names is None and hasattr(model, "classes_"):
        class_names = [str(c) for c in model.classes_]

    n_classes = int(model.n_classes_) if classifier else None

    root = _visit(
        tree=tree,
        node_id=0,
        classifier=classifier,
        n_classes=n_classes,
        feature_names=feature_names,
    )

    return TreeIR(
        root=root,
        n_features=int(model.n_features_in_),
        n_classes=n_classes,
        is_classifier=classifier,
        feature_names=feature_names,
        class_names=class_names,
        task="classification" if classifier else "regression",
        metadata={
            "criterion": getattr(model, "criterion", None),
            "max_depth": getattr(model, "max_depth", None),
            "sklearn_n_nodes": int(tree.node_count),
        },
    )


def _visit(
    tree: Any,
    node_id: int,
    classifier: bool,
    n_classes: int | None,
    feature_names: list[str] | None,
) -> TreeNode:
    left = tree.children_left[node_id]
    right = tree.children_right[node_id]

    # Leaf: sklearn uses -1 for missing children
    if left == -1 and right == -1:
        return _make_leaf(tree, node_id, classifier, n_classes)

    feature = int(tree.feature[node_id])
    threshold = float(tree.threshold[node_id])
    fname = None
    if feature_names is not None and 0 <= feature < len(feature_names):
        fname = feature_names[feature]

    return Node(
        feature=feature,
        threshold=threshold,
        left=_visit(tree, int(left), classifier, n_classes, feature_names),
        right=_visit(tree, int(right), classifier, n_classes, feature_names),
        feature_name=fname,
        n_samples=int(tree.n_node_samples[node_id]),
        node_id=node_id,
    )


def _make_leaf(
    tree: Any,
    node_id: int,
    classifier: bool,
    n_classes: int | None,
) -> Leaf:
    # tree.value shape: (n_nodes, n_outputs, n_classes_or_1)
    value = tree.value[node_id]

    if classifier:
        # value[0] is class counts / weighted counts
        counts = np.asarray(value[0], dtype=float)
        total = counts.sum()
        if total <= 0:
            prediction: int | float = 0
            probabilities = [0.0] * (n_classes or len(counts))
        else:
            prediction = int(np.argmax(counts))
            probabilities = (counts / total).tolist()
        return Leaf(
            prediction=prediction,
            probabilities=probabilities,
            n_samples=int(tree.n_node_samples[node_id]),
        )

    # Regressor: single continuous value
    prediction = float(value[0][0])
    return Leaf(
        prediction=prediction,
        probabilities=None,
        n_samples=int(tree.n_node_samples[node_id]),
    )
