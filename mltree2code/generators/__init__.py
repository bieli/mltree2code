from __future__ import annotations

from mltree2code.exceptions import UnsupportedLanguageError
from mltree2code.generators.base import BaseGenerator, EmitOptions
from mltree2code.generators.python import PythonGenerator

# Canonical language name → generator class
GENERATORS: dict[str, type[BaseGenerator]] = {
    "python": PythonGenerator,
    "py": PythonGenerator,
}

# Unique display names for help text
SUPPORTED_LANGUAGES = (
    "python",
)


def get_generator(language: str, options: EmitOptions | None = None) -> BaseGenerator:
    key = language.strip().lower()
    cls = GENERATORS.get(key)
    if cls is None:
        supported = ", ".join(SUPPORTED_LANGUAGES)
        raise UnsupportedLanguageError(
            f"Unsupported language '{language}'. Supported: {supported}"
        )
    return cls(options=options)


__all__ = [
    "GENERATORS",
    "SUPPORTED_LANGUAGES",
    "EmitOptions",
    "BaseGenerator",
    "get_generator",
]
