from __future__ import annotations

from pathlib import Path

import joblib
import pytest

from mltree2code.emitter import convert, convert_file
from mltree2code.exceptions import ModelLoadError, UnsupportedLanguageError
from mltree2code.generators.base import EmitOptions


def test_convert_returns_python_source(iris_model):
    model, _ = iris_model
    code = convert(model, "python")
    assert isinstance(code, str)
    assert code.endswith("\n")
    assert "def predict(x):" in code
    assert "if " in code
    assert "return " in code


def test_convert_language_aliases(iris_model):
    model, _ = iris_model
    assert "def predict" in convert(model, "py")
    assert "pub fn predict" in convert(model, "rs")
    assert "function predict" in convert(model, "js")


@pytest.mark.parametrize(
    "lang",
    ["python", "c", "cpp", "rust", "javascript", "java", "micropython"],
)
def test_convert_all_supported_languages(iris_model, lang):
    model, _ = iris_model
    code = convert(model, lang, options=EmitOptions(include_comments=False))
    assert "predict" in code
    assert len(code) > 10


def test_convert_respects_function_name(iris_model):
    model, _ = iris_model
    code = convert(model, "python", options=EmitOptions(function_name="score"))
    assert "def score(x):" in code
    assert "def predict(x):" not in code


def test_convert_no_comments(iris_model):
    model, _ = iris_model
    code = convert(model, "python", options=EmitOptions(include_comments=False))
    assert "Auto-generated" not in code
    assert code.lstrip().startswith("def predict")


def test_convert_with_feature_and_class_names(iris_model):
    model, _ = iris_model
    code = convert(
        model,
        "python",
        feature_names=["f0", "f1", "f2", "f3"],
        class_names=["a", "b", "c"],
        options=EmitOptions(include_comments=False),
    )
    assert "def predict(x):" in code


def test_convert_do_optimize_still_valid(iris_model):
    model, iris = iris_model
    code = convert(model, "python", do_optimize=True, options=EmitOptions(include_comments=False))
    ns: dict = {}
    exec(code, ns)  # noqa: S102
    predict = ns["predict"]
    for x in iris.data[:20]:
        assert predict(x) == int(model.predict([x])[0])


def test_convert_probabilities(iris_model):
    model, _ = iris_model
    code = convert(
        model,
        "python",
        options=EmitOptions(probabilities=True, include_comments=False),
    )
    assert "return [" in code


def test_convert_unsupported_language(iris_model):
    model, _ = iris_model
    with pytest.raises(UnsupportedLanguageError):
        convert(model, "brainfuck")


def test_convert_file_from_joblib(tmp_path: Path, iris_model):
    model, _ = iris_model
    path = tmp_path / "tree.joblib"
    joblib.dump(model, path)
    code = convert_file(path, "c", options=EmitOptions(include_comments=False))
    assert "int predict" in code
    assert code.endswith("\n")


def test_convert_file_accepts_str_path(tmp_path: Path, iris_model):
    model, _ = iris_model
    path = tmp_path / "tree.joblib"
    joblib.dump(model, path)
    code = convert_file(str(path), "rust")
    assert "pub fn predict" in code


def test_convert_file_with_options(tmp_path: Path, iris_model):
    model, _ = iris_model
    path = tmp_path / "tree.joblib"
    joblib.dump(model, path)
    code = convert_file(
        path,
        "python",
        options=EmitOptions(function_name="infer", include_comments=False),
        do_optimize=True,
    )
    assert "def infer(x):" in code


def test_convert_file_missing_path():
    with pytest.raises(ModelLoadError):
        convert_file("/nonexistent/model.joblib", "python")


def test_convert_normalize_newlines_single_trailing(iris_model):
    model, _ = iris_model
    code = convert(model, "python")
    assert not code.endswith("\n\n")
    assert code.endswith("\n")


def test_convert_regressor(regressor_model):
    model, _, _ = regressor_model
    code = convert(model, "python", options=EmitOptions(include_comments=False))
    assert "def predict(x):" in code
    assert "return " in code
