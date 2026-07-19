from __future__ import annotations

import numpy as np
import pytest

from mltree2code.emitter import convert
from mltree2code.generators import SUPPORTED_LANGUAGES
from mltree2code.generators.base import EmitOptions
from mltree2code.ir import Leaf, Node
from mltree2code.parser import parse_sklearn_tree
from tests.conftest import load_fixture


def _exec_python_predict(code: str):
    ns: dict = {}
    exec(code, ns)  # noqa: S102
    return ns["predict"]


def _max_if_nesting(code: str) -> int:
    max_depth = 0
    for line in code.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("if "):
            indent = len(line) - len(stripped)
            level = indent // 4
            max_depth = max(max_depth, level + 1)
    return max_depth


def test_iris_depth5_ir_has_depth_five():
    model = load_fixture("iris_depth5")
    assert model.max_depth == 5
    tree = parse_sklearn_tree(model)
    assert tree.depth() == 5
    assert tree.leaf_count() >= 2
    assert isinstance(tree.root, Node)


def test_iris_depth5_python_emits_nested_ifs():
    model = load_fixture("iris_depth5")
    code = convert(model, "python", options=EmitOptions(include_comments=False))
    assert "def predict(x):" in code
    # Root if is at indent level 1 -> nesting count includes deeper branches
    assert _max_if_nesting(code) >= 5
    assert code.count("if ") >= 5


@pytest.mark.parametrize("lang", SUPPORTED_LANGUAGES)
def test_iris_depth5_all_languages_emit(lang):
    model = load_fixture("iris_depth5")
    code = convert(model, lang, options=EmitOptions(include_comments=False))
    assert "predict" in code
    assert "if" in code.lower()
    assert len(code) > 50


def test_iris_depth5_python_matches_sklearn():
    model = load_fixture("iris_depth5")
    code = convert(model, "python", options=EmitOptions(include_comments=False))
    predict = _exec_python_predict(code)

    n = model.n_features_in_
    rng = np.random.default_rng(42)
    X = np.vstack(
        [
            rng.normal(size=(80, n)),
            np.zeros((1, n)),
            np.ones((1, n)),
            rng.uniform(-2, 8, size=(40, n)),
        ]
    )
    for x in X:
        assert int(predict(x)) == int(model.predict([x])[0])


def test_iris_depth5_walk_reaches_deep_leaves():
    model = load_fixture("iris_depth5")
    tree = parse_sklearn_tree(model)

    def path_depth(node, depth=0) -> int:
        if isinstance(node, Leaf):
            return depth
        return max(path_depth(node.left, depth + 1), path_depth(node.right, depth + 1))

    assert path_depth(tree.root) == 5
