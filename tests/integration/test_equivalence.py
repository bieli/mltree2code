from __future__ import annotations

import numpy as np
import pytest

from mltree2code.emitter import convert
from mltree2code.generators import SUPPORTED_LANGUAGES
from tests.conftest import load_fixture
from tests.model_fixtures import MODEL_FIXTURES


def _exec_python_predict(code: str):
    ns: dict = {}
    exec(code, ns)  # noqa: S102 - intentional for testing generated code
    return ns["predict"]


def test_python_matches_sklearn_iris(iris_model):
    model, iris = iris_model
    code = convert(model, "python")
    predict = _exec_python_predict(code)
    for x in iris.data:
        assert predict(x) == int(model.predict([x])[0])


def test_micropython_matches_sklearn(iris_model):
    model, iris = iris_model
    code = convert(model, "micropython")
    predict = _exec_python_predict(code)
    for x in iris.data[:30]:
        assert predict(list(x)) == int(model.predict([x])[0])


@pytest.mark.parametrize("fixture", MODEL_FIXTURES)
def test_fixture_python_equivalence(fixture):
    model = load_fixture(fixture)
    code = convert(model, "python")
    predict = _exec_python_predict(code)
    n = model.n_features_in_
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, n))
    X = np.vstack([X, np.zeros((1, n)), np.ones((1, n))])
    for x in X:
        expected = model.predict([x])[0]
        got = predict(x)
        if hasattr(expected, "item"):
            expected = expected.item()
        if isinstance(expected, (np.floating, float)):
            assert got == pytest.approx(float(expected), rel=1e-5, abs=1e-5)
        else:
            assert int(got) == int(expected)


def test_regressor_equivalence(regressor_model):
    model, X, _ = regressor_model
    code = convert(model, "python")
    predict = _exec_python_predict(code)
    for x in X:
        assert predict(x) == pytest.approx(float(model.predict([x])[0]), rel=1e-6)


def test_all_languages_nonempty_for_fixtures():
    model = load_fixture("iris_depth2")
    for lang in SUPPORTED_LANGUAGES:
        code = convert(model, lang)
        assert "if" in code.lower() or "return" in code.lower()


def test_optimize_preserves_semantics(iris_model):
    model, iris = iris_model
    code = convert(model, "python", do_optimize=True)
    predict = _exec_python_predict(code)
    for x in iris.data:
        assert predict(x) == int(model.predict([x])[0])
