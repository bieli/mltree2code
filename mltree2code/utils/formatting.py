from __future__ import annotations


def indent_str(level: int, size: int = 4, use_tabs: bool = False) -> str:
    if use_tabs:
        return "\t" * level
    return " " * (size * level)


def format_float(value: float, precision: int = 17, float_suffix: str = "") -> str:
    """Format floats for source emission with enough digits for float64 round-trip.

    Using too few significant digits (e.g. 6) can change ``<=`` decisions when a
    feature value sits on a sklearn threshold boundary.
    """
    # Cap at 17 significant digits — enough for IEEE-754 binary64 round-trips.
    digits = max(1, min(precision, 17))
    formatted = format(float(value), f".{digits}g")
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
