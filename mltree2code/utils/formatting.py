from __future__ import annotations


def indent_str(level: int, size: int = 4, use_tabs: bool = False) -> str:
    if use_tabs:
        return "\t" * level
    return " " * (size * level)


def format_float(value: float, precision: int = 6, float_suffix: str = "") -> str:
    formatted = f"{value:.{precision}g}"
    if "." not in formatted and "e" not in formatted.lower():
        formatted += ".0"
    return formatted + float_suffix


def feature_access(
    feature: int,
    *,
    style: str = "index",
    feature_names: list[str] | None = None,
    array_name: str = "x",
) -> str:
    if style == "name" and feature_names and 0 <= feature < len(feature_names):
        name = feature_names[feature]
        if name.isidentifier():
            return name
    return f"{array_name}[{feature}]"
