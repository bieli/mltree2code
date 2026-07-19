# Extending with a new generator

1. Create `mltree2code/generators/mylang.py` subclassing `BaseGenerator`.
2. Implement `emit_function`, `emit_node`, and `emit_leaf`.
3. Register aliases in `mltree2code/generators/__init__.py` (`GENERATORS` +
   `SUPPORTED_LANGUAGES`).
4. Add a unit test that checks a structural snippet.
5. Ensure semantic equivalence still passes for the Python path (logic is shared
   via IR).

Minimal skeleton:

```python
from mltree2code.generators.base import BaseGenerator
from mltree2code.ir import Leaf, Node, TreeIR

class GoGenerator(BaseGenerator):
    language = "go"
    file_extension = ".go"

    def emit_function(self, tree: TreeIR) -> None:
        self.add(f"func {self.options.function_name}(x []float64) int {{")
        self.visit(tree.root, tree, 1)
        self.add("}")

    def emit_leaf(self, leaf: Leaf, tree: TreeIR, level: int) -> None:
        self.add(f"{self.ind(level)}return {int(leaf.prediction)}")

    def emit_node(self, node: Node, tree: TreeIR, level: int) -> None:
        ind = self.ind(level)
        self.add(f"{ind}if {self.feat(node.feature)} <= {self.fmt_float(node.threshold)} {{")
        self.visit(node.left, tree, level + 1)
        self.add(f"{ind}}} else {{")
        self.visit(node.right, tree, level + 1)
        self.add(f"{ind}}}")
```
