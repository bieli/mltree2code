from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union


@dataclass
class Leaf:
    prediction: int | float
    probabilities: list[float] | None = None
    n_samples: int | None = None

    def is_leaf(self) -> bool:
        return True


@dataclass
class Node:
    feature: int
    threshold: float
    left: TreeNode
    right: TreeNode
    feature_name: str | None = None
    n_samples: int | None = None
    node_id: int | None = None

    def is_leaf(self) -> bool:
        return False


TreeNode = Union[Node, Leaf]


@dataclass
class TreeIR:
    root: TreeNode
    n_features: int
    n_classes: int | None = None  # None for regressors
    is_classifier: bool = True
    feature_names: list[str] | None = None
    class_names: list[str] | None = None
    task: str = "classification"  # or "regression"
    metadata: dict = field(default_factory=dict)

    def walk(self):
        stack: list[TreeNode] = [self.root]
        while stack:
            node = stack.pop()
            yield node
            if isinstance(node, Node):
                stack.append(node.right)
                stack.append(node.left)

    def leaf_count(self) -> int:
        return sum(1 for n in self.walk() if isinstance(n, Leaf))

    def depth(self) -> int:
        def _depth(node: TreeNode) -> int:
            if isinstance(node, Leaf):
                return 0
            return 1 + max(_depth(node.left), _depth(node.right))

        return _depth(self.root)
