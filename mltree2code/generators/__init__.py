from __future__ import annotations

from mltree2code.exceptions import UnsupportedLanguageError
from mltree2code.generators.base import BaseGenerator, EmitOptions
from mltree2code.generators.c import CGenerator
from mltree2code.generators.cpp import CppGenerator
from mltree2code.generators.java import JavaGenerator
from mltree2code.generators.javascript import JavaScriptGenerator
from mltree2code.generators.micropython import MicroPythonGenerator
from mltree2code.generators.python import PythonGenerator
from mltree2code.generators.rust import RustGenerator

GENERATORS: dict[str, type[BaseGenerator]] = {
    "python": PythonGenerator,
    "py": PythonGenerator,
    "micropython": MicroPythonGenerator,
    "upy": MicroPythonGenerator,
    "c": CGenerator,
    "cpp": CppGenerator,
    "c++": CppGenerator,
    "cxx": CppGenerator,
    "rust": RustGenerator,
    "rs": RustGenerator,
    "javascript": JavaScriptGenerator,
    "js": JavaScriptGenerator,
    "node": JavaScriptGenerator,
    "java": JavaGenerator,
}

SUPPORTED_LANGUAGES = (
    "python",
    "micropython",
    "c",
    "cpp",
    "rust",
    "javascript",
    "java",
)


def get_generator(language: str, options: EmitOptions | None = None) -> BaseGenerator:
    key = language.strip().lower()
    cls = GENERATORS.get(key)
    if cls is None:
        supported = ", ".join(SUPPORTED_LANGUAGES)
        raise UnsupportedLanguageError(f"Unsupported language '{language}'. Supported: {supported}")
    return cls(options=options)


__all__ = [
    "GENERATORS",
    "SUPPORTED_LANGUAGES",
    "EmitOptions",
    "BaseGenerator",
    "get_generator",
]
