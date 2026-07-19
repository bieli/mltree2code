from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from mltree2code.ir import Leaf, Node, TreeIR, TreeNode
from mltree2code.utils.formatting import feature_access, format_float, indent_str


@dataclass
class EmitOptions:
    function_name: str = "predict"
    indent: int = 4
    use_tabs: bool = False
    float_precision: int = 6
    float_type: str = "float"
    class_names: list[str] | None = None
    feature_names: list[str] | None = None
    use_feature_names: bool = False
    probabilities: bool = False
    namespace: str | None = None
    header: bool = False
    include_comments: bool = True
    extra: dict = field(default_factory=dict)


class BaseGenerator(ABC):
    language: str = "base"
    file_extension: str = ".txt"

    def __init__(self, options: EmitOptions | None = None):
        self.options = options or EmitOptions()
        self.lines: list[str] = []

    def emit(self, tree: TreeIR) -> str:
        self.lines = []
        if self.options.feature_names is None:
            self.options.feature_names = tree.feature_names
        if self.options.class_names is None:
            self.options.class_names = tree.class_names
        self.emit_preamble(tree)
        self.emit_function(tree)
        self.emit_epilogue(tree)
        return "\n".join(self.lines).rstrip() + "\n"

    def emit_preamble(self, tree: TreeIR) -> None:
        return

    def emit_epilogue(self, tree: TreeIR) -> None:
        return

    @abstractmethod
    def emit_function(self, tree: TreeIR) -> None:
        return

    def ind(self, level: int) -> str:
        return indent_str(level, self.options.indent, self.options.use_tabs)

    def fmt_float(self, value: float, suffix: str = "") -> str:
        return format_float(value, self.options.float_precision, suffix)

    def feat(self, feature: int, array_name: str = "x") -> str:
        style = "name" if self.options.use_feature_names else "index"
        return feature_access(
            feature,
            style=style,
            feature_names=self.options.feature_names,
            array_name=array_name,
        )

    def prediction_value(self, leaf: Leaf, tree: TreeIR):
        if tree.is_classifier and self.options.probabilities and leaf.probabilities:
            return leaf.probabilities
        return leaf.prediction

    def add(self, line: str = "") -> None:
        self.lines.append(line)

    def visit(self, node: TreeNode, tree: TreeIR, level: int) -> None:
        if isinstance(node, Leaf):
            self.emit_leaf(node, tree, level)
        else:
            assert isinstance(node, Node)
            self.emit_node(node, tree, level)

    def emit_leaf(self, leaf: Leaf, tree: TreeIR, level: int) -> None:
        raise NotImplementedError

    def emit_node(self, node: Node, tree: TreeIR, level: int) -> None:
        raise NotImplementedError
