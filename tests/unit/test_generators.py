from __future__ import annotations

import pytest

from mltree2code.emitter import convert
from mltree2code.exceptions import UnsupportedLanguageError
from mltree2code.generators import SUPPORTED_LANGUAGES, get_generator
from mltree2code.generators.base import EmitOptions


@pytest.mark.parametrize("lang", SUPPORTED_LANGUAGES)
def test_all_languages_emit(iris_model, lang):
    model, _ = iris_model
    code = convert(model, lang)
    assert "predict" in code
    assert len(code) > 20


def test_python_structure(iris_model):
    model, _ = iris_model
    code = convert(model, "python")
    assert "def predict(x):" in code
    assert "if " in code
    assert "return " in code


def test_c_structure(iris_model):
    model, _ = iris_model
    code = convert(model, "c")
    assert "int predict(const float x[])" in code
    assert "return " in code


def test_rust_structure(iris_model):
    model, _ = iris_model
    code = convert(model, "rust")
    assert "pub fn predict(x: &[f32]) -> i32" in code


def test_java_namespace(iris_model):
    model, _ = iris_model
    opts = EmitOptions(namespace="com.example.ml", extra={"class_name": "IrisTree"})
    code = convert(model, "java", options=opts)
    assert "package com.example.ml;" in code
    assert "public final class IrisTree" in code


def test_function_name(iris_model):
    model, _ = iris_model
    opts = EmitOptions(function_name="score")
    code = convert(model, "python", options=opts)
    assert "def score(x):" in code


def test_unsupported_language(iris_model):
    model, _ = iris_model
    with pytest.raises(UnsupportedLanguageError):
        get_generator("cobol")


def test_regressor_python(regressor_model):
    model, _, _ = regressor_model
    code = convert(model, "python")
    assert "def predict(x):" in code
    assert "return " in code


def test_probabilities(iris_model):
    model, _ = iris_model
    opts = EmitOptions(probabilities=True)
    code = convert(model, "python", options=opts)
    assert "return [" in code
