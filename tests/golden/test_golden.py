"""
Golden snapshot tests for generated code formatting

Run with UPDATE_GOLDEN=1 to refresh snapshots after intentional changes:
$ UPDATE_GOLDEN=1 pytest tests/golden -q
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from mltree2code.emitter import convert
from mltree2code.generators import SUPPORTED_LANGUAGES
from mltree2code.generators.base import EmitOptions
from tests.conftest import load_fixture

GOLDEN_DIR = Path(__file__).parent / "snapshots"


def _golden_path(fixture: str, lang: str) -> Path:
    return GOLDEN_DIR / fixture / f"{lang}.txt"


@pytest.mark.parametrize("lang", SUPPORTED_LANGUAGES)
def test_golden_iris_depth2(lang):
    model = load_fixture("iris_depth2")
    opts = EmitOptions(include_comments=False, function_name="predict")
    code = convert(model, lang, options=opts)
    path = _golden_path("iris_depth2", lang)
    if os.environ.get("UPDATE_GOLDEN") == "1":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(code, encoding="utf-8")
        pytest.skip("golden updated")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(code, encoding="utf-8")
        pytest.skip(f"created golden {path}")
    assert code == path.read_text(encoding="utf-8")
