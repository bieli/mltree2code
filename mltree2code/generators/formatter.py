from __future__ import annotations


def normalize_newlines(code: str) -> str:
    code = code.replace("\r\n", "\n").replace("\r", "\n")
    if not code.endswith("\n"):
        code += "\n"
    return code

