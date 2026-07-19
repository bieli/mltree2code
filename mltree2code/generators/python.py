from __future__ import annotations

from mltree2code.generators.base import BaseGenerator
from mltree2code.ir import Leaf, Node, TreeIR


class PythonGenerator(BaseGenerator):
    language = "python"
    file_extension = ".py"

    def emit_preamble(self, tree: TreeIR) -> None:
        raise NotImplementedError

    def emit_function(self, tree: TreeIR) -> None:
        raise NotImplementedError

    def emit_leaf(self, leaf: Leaf, tree: TreeIR, level: int) -> None:
        raise NotImplementedError

    def emit_node(self, node: Node, tree: TreeIR, level: int) -> None:
        raise NotImplementedError

