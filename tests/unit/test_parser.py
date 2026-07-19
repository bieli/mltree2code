from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

from mltree2code.exceptions import ParseError
from mltree2code.ir import Leaf, Node, TreeIR
from mltree2code.parser import _make_leaf, _visit, parse_sklearn_tree


def _stump_classifier_depth1_binary_stump_with_known_split():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    clf = DecisionTreeClassifier(max_depth=1, random_state=0)
    clf.fit(X, y)
    return clf, X, y


def test_parse_raises_without_tree_attribute():
    with pytest.raises(ParseError, match="tree_"):
        parse_sklearn_tree(object())


def test_parse_raises_for_unfitted_estimator():
    with pytest.raises(ParseError, match="tree_"):
        parse_sklearn_tree(DecisionTreeClassifier())


def test_parse_classifier_returns_tree_ir(iris_model):
    model, iris = iris_model
    tree = parse_sklearn_tree(model)
    assert isinstance(tree, TreeIR)
    assert tree.is_classifier is True
    assert tree.task == "classification"
    assert tree.n_features == 4
    assert tree.n_classes == 3
    assert tree.metadata["sklearn_n_nodes"] == model.tree_.node_count
    assert tree.metadata["criterion"] == model.criterion
    assert tree.metadata["max_depth"] == model.max_depth


def test_parse_uses_model_feature_names_in_when_present():
    iris = load_iris()
    clf = DecisionTreeClassifier(max_depth=2, random_state=0)
    clf.fit(iris.data, iris.target)
    # Simulate estimators fitted with named columns (no pandas dependency).
    clf.feature_names_in_ = np.array(
        ["sepal_length", "sepal_width", "petal_length", "petal_width"],
        dtype=object,
    )
    tree = parse_sklearn_tree(clf)
    assert tree.feature_names == [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
    ]


def test_parse_uses_model_classes_when_present(iris_model):
    model, _ = iris_model
    tree = parse_sklearn_tree(model)
    assert tree.class_names == [str(c) for c in model.classes_]


def test_parse_explicit_names_override_model(iris_model):
    model, _ = iris_model
    feats = ["a", "b", "c", "d"]
    classes = ["x", "y", "z"]
    tree = parse_sklearn_tree(model, feature_names=feats, class_names=classes)
    assert tree.feature_names == feats
    assert tree.class_names == classes


def test_parse_attaches_feature_name_on_nodes():
    clf, _, _ = _stump_classifier_depth1_binary_stump_with_known_split()
    tree = parse_sklearn_tree(clf, feature_names=["value"])
    assert isinstance(tree.root, Node)
    assert tree.root.feature == 0
    assert tree.root.feature_name == "value"
    assert tree.root.node_id == 0
    assert tree.root.n_samples == 4


def test_parse_stump_structure_and_thresholds():
    clf, _, _ = _stump_classifier_depth1_binary_stump_with_known_split()
    tree = parse_sklearn_tree(clf)
    root = tree.root
    assert isinstance(root, Node)
    assert isinstance(root.left, Leaf)
    assert isinstance(root.right, Leaf)
    assert root.left.prediction == 0
    assert root.right.prediction == 1
    assert 1.0 <= root.threshold <= 2.0


def test_parse_leaf_probabilities_sum_to_one(iris_model):
    model, _ = iris_model
    tree = parse_sklearn_tree(model)
    for node in tree.walk():
        if isinstance(node, Leaf):
            assert node.probabilities is not None
            assert len(node.probabilities) == tree.n_classes
            assert pytest.approx(sum(node.probabilities), abs=1e-9) == 1.0
            assert node.prediction == int(np.argmax(node.probabilities))


def test_parse_regressor_leaf_has_float_prediction(regressor_model):
    model, _, _ = regressor_model
    tree = parse_sklearn_tree(model)
    assert tree.is_classifier is False
    assert tree.task == "regression"
    assert tree.n_classes is None
    assert tree.class_names is None
    leaves = [n for n in tree.walk() if isinstance(n, Leaf)]
    assert leaves
    for leaf in leaves:
        assert isinstance(leaf.prediction, float)
        assert leaf.probabilities is None


def test_parse_matches_sklearn_tree_arrays(iris_model):
    model, _ = iris_model
    sk = model.tree_
    tree = parse_sklearn_tree(model)

    def check(node, node_id: int) -> None:
        left = sk.children_left[node_id]
        right = sk.children_right[node_id]
        if left == -1 and right == -1:
            assert isinstance(node, Leaf)
            counts = np.asarray(sk.value[node_id][0], dtype=float)
            assert node.prediction == int(np.argmax(counts))
            return
        assert isinstance(node, Node)
        assert node.feature == int(sk.feature[node_id])
        assert node.threshold == pytest.approx(float(sk.threshold[node_id]))
        assert node.node_id == node_id
        check(node.left, int(left))
        check(node.right, int(right))

    check(tree.root, 0)


def test_visit_builds_leaf_for_terminal_node():
    clf, _, _ = _stump_classifier_depth1_binary_stump_with_known_split()
    sk = clf.tree_
    leaf_id = next(i for i, c in enumerate(sk.children_left) if c == -1)
    node = _visit(sk, leaf_id, classifier=True, n_classes=2, feature_names=None)
    assert isinstance(node, Leaf)


def test_make_leaf_classifier_zero_total_counts():
    class FakeTree:
        value = {0: np.array([[0.0, 0.0]])}
        n_node_samples = {0: 0}

    leaf = _make_leaf(FakeTree(), 0, classifier=True, n_classes=2)
    assert leaf.prediction == 0
    assert leaf.probabilities == [0.0, 0.0]
    assert leaf.n_samples == 0


def test_make_leaf_regressor():
    class FakeTree:
        value = {0: np.array([[3.14]])}
        n_node_samples = {0: 7}

    leaf = _make_leaf(FakeTree(), 0, classifier=False, n_classes=None)
    assert leaf.prediction == pytest.approx(3.14)
    assert leaf.probabilities is None
    assert leaf.n_samples == 7


def test_feature_name_out_of_range_is_none():
    clf, _, _ = _stump_classifier_depth1_binary_stump_with_known_split()
    tree = parse_sklearn_tree(clf, feature_names=[])
    assert isinstance(tree.root, Node)
    assert tree.root.feature_name is None


def test_pure_leaf_tree_when_constant_target():
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([1, 1, 1])
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X, y)
    tree = parse_sklearn_tree(clf)
    assert isinstance(tree.root, Leaf)
    # prediction is the class index (0), not the raw label
    assert tree.root.prediction == 0
    assert tree.class_names == ["1"]
    assert tree.depth() == 0
    assert tree.leaf_count() == 1
