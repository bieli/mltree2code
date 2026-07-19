from __future__ import annotations

from copy import deepcopy

from mltree2code.ir import Leaf, Node, TreeIR, TreeNode


def optimize(tree: TreeIR, *, merge_identical_leaves: bool = True) -> TreeIR:
    result = deepcopy(tree)
    if merge_identical_leaves:
        result.root = _merge_identical(result.root)
    return result


def _leaves_equal(a: Leaf, b: Leaf) -> bool:
    if a.prediction != b.prediction:
        return False
    if a.probabilities is None and b.probabilities is None:
        return True
    if a.probabilities is None or b.probabilities is None:
        return False
    if len(a.probabilities) != len(b.probabilities):
        return False
    return all(abs(x - y) < 1e-12 for x, y in zip(a.probabilities, b.probabilities))


def _merge_identical(node: TreeNode) -> TreeNode:
    if isinstance(node, Leaf):
        return node

    left = _merge_identical(node.left)
    right = _merge_identical(node.right)

    if isinstance(left, Leaf) and isinstance(right, Leaf) and _leaves_equal(left, right):
        n_samples = None
        if left.n_samples is not None and right.n_samples is not None:
            n_samples = left.n_samples + right.n_samples
        return Leaf(
            prediction=left.prediction,
            probabilities=left.probabilities,
            n_samples=n_samples,
        )

    return Node(
        feature=node.feature,
        threshold=node.threshold,
        left=left,
        right=right,
        feature_name=node.feature_name,
        n_samples=node.n_samples,
        node_id=node.node_id,
    )
