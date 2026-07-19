from __future__ import annotations

import copy

import pytest

from mltree2code.emitter import convert
from mltree2code.generators.base import EmitOptions
from mltree2code.ir import Leaf, Node, TreeIR
from mltree2code.optimizer import _leaves_equal, optimize
from mltree2code.parser import parse_sklearn_tree


def _clf_tree(root) -> TreeIR:
    return TreeIR(root=root, n_features=2, n_classes=2, is_classifier=True)


def _reg_tree(root) -> TreeIR:
    return TreeIR(
        root=root,
        n_features=1,
        n_classes=None,
        is_classifier=False,
        task="regression",
    )


def test_merge_identical_leaves():
    root = Node(
        feature=0,
        threshold=0.5,
        left=Leaf(prediction=1, probabilities=[0.0, 1.0]),
        right=Leaf(prediction=1, probabilities=[0.0, 1.0]),
    )
    opt = optimize(_clf_tree(root))
    assert isinstance(opt.root, Leaf)
    assert opt.root.prediction == 1
    assert opt.root.probabilities == [0.0, 1.0]


def test_no_merge_different_predictions():
    root = Node(
        feature=0,
        threshold=0.5,
        left=Leaf(prediction=0),
        right=Leaf(prediction=1),
    )
    opt = optimize(_clf_tree(root))
    assert isinstance(opt.root, Node)
    assert opt.root.left.prediction == 0
    assert opt.root.right.prediction == 1


def test_optimize_does_not_mutate_input():
    left = Leaf(prediction=1, n_samples=3)
    right = Leaf(prediction=1, n_samples=4)
    root = Node(feature=0, threshold=0.5, left=left, right=right, node_id=0)
    tree = _clf_tree(root)
    before = copy.deepcopy(tree)
    opt = optimize(tree)
    assert isinstance(opt.root, Leaf)
    assert isinstance(tree.root, Node)
    assert tree.root is root
    assert tree.root.left is left
    assert tree == before


def test_merge_disabled_is_noop():
    root = Node(
        feature=0,
        threshold=0.5,
        left=Leaf(prediction=1),
        right=Leaf(prediction=1),
    )
    tree = _clf_tree(root)
    opt = optimize(tree, merge_identical_leaves=False)
    assert isinstance(opt.root, Node)
    assert opt.depth() == 1
    assert opt.leaf_count() == 2


def test_already_leaf_is_identity():
    tree = _clf_tree(Leaf(prediction=0, probabilities=[1.0, 0.0], n_samples=10))
    opt = optimize(tree)
    assert isinstance(opt.root, Leaf)
    assert opt.root.prediction == 0
    assert opt.depth() == 0


def test_cascading_collapse_to_single_leaf():
    # Deep tree where every leaf is class 1 -> whole tree becomes one leaf
    #        N0
    #       /  \
    #     N1    L1
    #    /  \
    #  L1   L1
    root = Node(
        feature=0,
        threshold=0.5,
        left=Node(
            feature=1,
            threshold=1.0,
            left=Leaf(prediction=1, probabilities=[0.0, 1.0], n_samples=2),
            right=Leaf(prediction=1, probabilities=[0.0, 1.0], n_samples=3),
            node_id=1,
        ),
        right=Leaf(prediction=1, probabilities=[0.0, 1.0], n_samples=5),
        node_id=0,
    )
    opt = optimize(_clf_tree(root))
    assert isinstance(opt.root, Leaf)
    assert opt.root.prediction == 1
    assert opt.root.n_samples == 2 + 3 + 5
    assert opt.depth() == 0
    assert opt.leaf_count() == 1


def test_partial_merge_keeps_parent_split():
    # Only the left subtree collapses; root still splits
    #        N0
    #       /  \
    #     N1    L0
    #    /  \
    #  L1   L1
    root = Node(
        feature=0,
        threshold=0.5,
        feature_name="x0",
        node_id=0,
        n_samples=20,
        left=Node(
            feature=1,
            threshold=2.0,
            left=Leaf(prediction=1, n_samples=4),
            right=Leaf(prediction=1, n_samples=6),
            node_id=1,
        ),
        right=Leaf(prediction=0, n_samples=10),
    )
    opt = optimize(_clf_tree(root))
    assert isinstance(opt.root, Node)
    assert opt.root.feature == 0
    assert opt.root.feature_name == "x0"
    assert opt.root.node_id == 0
    assert opt.root.threshold == 0.5
    assert isinstance(opt.root.left, Leaf)
    assert opt.root.left.prediction == 1
    assert opt.root.left.n_samples == 10
    assert isinstance(opt.root.right, Leaf)
    assert opt.root.right.prediction == 0
    assert opt.depth() == 1
    assert opt.leaf_count() == 2


def test_n_samples_summed_when_both_present():
    root = Node(
        feature=0,
        threshold=0.5,
        left=Leaf(prediction=0, n_samples=7),
        right=Leaf(prediction=0, n_samples=11),
    )
    opt = optimize(_clf_tree(root))
    assert isinstance(opt.root, Leaf)
    assert opt.root.n_samples == 18


def test_n_samples_none_if_either_missing():
    root = Node(
        feature=0,
        threshold=0.5,
        left=Leaf(prediction=0, n_samples=7),
        right=Leaf(prediction=0, n_samples=None),
    )
    opt = optimize(_clf_tree(root))
    assert isinstance(opt.root, Leaf)
    assert opt.root.n_samples is None


def test_no_merge_when_probabilities_differ():
    root = Node(
        feature=0,
        threshold=0.5,
        left=Leaf(prediction=1, probabilities=[0.1, 0.9]),
        right=Leaf(prediction=1, probabilities=[0.2, 0.8]),
    )
    opt = optimize(_clf_tree(root))
    assert isinstance(opt.root, Node)


def test_merge_when_probabilities_within_epsilon():
    root = Node(
        feature=0,
        threshold=0.5,
        left=Leaf(prediction=0, probabilities=[0.5, 0.5]),
        right=Leaf(prediction=0, probabilities=[0.5 + 5e-13, 0.5 - 5e-13]),
    )
    opt = optimize(_clf_tree(root))
    assert isinstance(opt.root, Leaf)


def test_no_merge_when_probabilities_outside_epsilon():
    root = Node(
        feature=0,
        threshold=0.5,
        left=Leaf(prediction=0, probabilities=[0.5, 0.5]),
        right=Leaf(prediction=0, probabilities=[0.5 + 1e-9, 0.5 - 1e-9]),
    )
    opt = optimize(_clf_tree(root))
    assert isinstance(opt.root, Node)


def test_no_merge_when_one_side_lacks_probabilities():
    root = Node(
        feature=0,
        threshold=0.5,
        left=Leaf(prediction=1, probabilities=[0.0, 1.0]),
        right=Leaf(prediction=1, probabilities=None),
    )
    opt = optimize(_clf_tree(root))
    assert isinstance(opt.root, Node)


def test_no_merge_when_probability_vector_lengths_differ():
    root = Node(
        feature=0,
        threshold=0.5,
        left=Leaf(prediction=0, probabilities=[1.0, 0.0]),
        right=Leaf(prediction=0, probabilities=[1.0, 0.0, 0.0]),
    )
    opt = optimize(_clf_tree(root))
    assert isinstance(opt.root, Node)


def test_regressor_merges_identical_values():
    root = Node(
        feature=0,
        threshold=1.5,
        left=Leaf(prediction=3.14, n_samples=2),
        right=Leaf(prediction=3.14, n_samples=5),
    )
    opt = optimize(_reg_tree(root))
    assert isinstance(opt.root, Leaf)
    assert opt.root.prediction == pytest.approx(3.14)
    assert opt.root.probabilities is None
    assert opt.root.n_samples == 7


def test_regressor_keeps_split_for_different_values():
    root = Node(
        feature=0,
        threshold=1.5,
        left=Leaf(prediction=1.0),
        right=Leaf(prediction=2.0),
    )
    opt = optimize(_reg_tree(root))
    assert isinstance(opt.root, Node)


def test_leaves_equal_helper():
    assert _leaves_equal(Leaf(1), Leaf(1))
    assert not _leaves_equal(Leaf(0), Leaf(1))
    assert _leaves_equal(
        Leaf(0, probabilities=[0.25, 0.75]),
        Leaf(0, probabilities=[0.25, 0.75]),
    )
    assert not _leaves_equal(
        Leaf(0, probabilities=[1.0, 0.0]),
        Leaf(0, probabilities=None),
    )


def test_metadata_and_task_preserved():
    tree = TreeIR(
        root=Node(
            feature=0,
            threshold=0.5,
            left=Leaf(prediction=0),
            right=Leaf(prediction=0),
        ),
        n_features=3,
        n_classes=2,
        is_classifier=True,
        feature_names=["a", "b", "c"],
        class_names=["no", "yes"],
        task="classification",
        metadata={"source": "unit"},
    )
    opt = optimize(tree)
    assert opt.n_features == 3
    assert opt.n_classes == 2
    assert opt.feature_names == ["a", "b", "c"]
    assert opt.class_names == ["no", "yes"]
    assert opt.metadata == {"source": "unit"}
    assert opt.is_classifier is True


def test_optimize_preserves_sklearn_predictions(iris_model):
    model, iris = iris_model
    plain = convert(model, "python", options=EmitOptions(include_comments=False))
    optimized = convert(
        model,
        "python",
        do_optimize=True,
        options=EmitOptions(include_comments=False),
    )
    ns_plain: dict = {}
    ns_opt: dict = {}
    exec(plain, ns_plain)  # noqa: S102
    exec(optimized, ns_opt)  # noqa: S102
    for x in iris.data:
        assert ns_plain["predict"](x) == ns_opt["predict"](x) == int(model.predict([x])[0])


def test_optimize_can_shrink_iris_ir(iris_model):
    model, _ = iris_model
    tree = parse_sklearn_tree(model)
    opt = optimize(tree)
    assert opt.leaf_count() <= tree.leaf_count()
    assert opt.depth() <= tree.depth()
    for node in opt.walk():
        if isinstance(node, Leaf):
            assert 0 <= int(node.prediction) < (opt.n_classes or 1)
