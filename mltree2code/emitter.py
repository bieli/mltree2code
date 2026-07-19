from __future__ import annotations

from pathlib import Path
from typing import Any, Union

from mltree2code.generators import get_generator
from mltree2code.generators.base import EmitOptions
from mltree2code.generators.formatter import normalize_newlines
from mltree2code.loader import load_model
from mltree2code.optimizer import optimize
from mltree2code.parser import parse_sklearn_tree

PathLike = Union[str, Path]


def convert(
    model: Any,
    language: str,
    *,
    options: EmitOptions | None = None,
    feature_names: list[str] | None = None,
    class_names: list[str] | None = None,
    do_optimize: bool = False,
) -> str:
    tree = parse_sklearn_tree(
        model,
        feature_names=feature_names,
        class_names=class_names,
    )
    if do_optimize:
        tree = optimize(tree)
    generator = get_generator(language, options=options)
    return normalize_newlines(generator.emit(tree))


def convert_file(
    model_path: PathLike,
    language: str,
    *,
    options: EmitOptions | None = None,
    feature_names: list[str] | None = None,
    class_names: list[str] | None = None,
    do_optimize: bool = False,
) -> str:
    model = load_model(model_path)
    return convert(
        model,
        language,
        options=options,
        feature_names=feature_names,
        class_names=class_names,
        do_optimize=do_optimize,
    )
