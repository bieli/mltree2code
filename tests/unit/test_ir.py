from __future__ import annotations

from mltree2code.ir import Leaf, Node, TreeIR
from mltree2code.parser import parse_sklearn_tree


def test_parse_iris_classifier(iris_model):
    model, iris = iris_model
    tree = parse_sklearn_tree(
        model,
        feature_names=list(iris.feature_names),
        class_names=list(iris.target_names),
    )
    assert isinstance(tree, TreeIR)
    assert tree.is_classifier
    assert tree.n_features == 4
    assert tree.n_classes == 3
    assert tree.depth() >= 1
    assert tree.leaf_count() >= 2
    assert tree.feature_names is not None
    assert tree.class_names is not None


def test_parse_regressor(regressor_model):
    model, X, y = regressor_model
    tree = parse_sklearn_tree(model)
    assert not tree.is_classifier
    assert tree.task == "regression"
    assert tree.n_classes is None
    assert isinstance(tree.root, (Node, Leaf))


def test_ir_walk(iris_model):
    model, _ = iris_model
    tree = parse_sklearn_tree(model)
    nodes = list(tree.walk())
    assert len(nodes) == tree.leaf_count() + sum(
        1 for n in nodes if isinstance(n, Node)
    )
    leaves = [n for n in nodes if isinstance(n, Leaf)]
    assert all(isinstance(leaf.prediction, int) for leaf in leaves)
