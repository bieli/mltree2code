from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from mltree2code import __version__
from mltree2code.emitter import convert_file
from mltree2code.exceptions import Mltree2codeError
from mltree2code.generators import SUPPORTED_LANGUAGES
from mltree2code.generators.base import EmitOptions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mltree2code",
        description=(
            "Transpile ML decision trees into readable if-else source code. "
            "Supported languages: " + ", ".join(SUPPORTED_LANGUAGES)
        ),
    )
    parser.add_argument(
        "model",
        nargs="?",
        help="Path to a pickled/joblib sklearn DecisionTree model",
    )
    parser.add_argument(
        "language",
        nargs="?",
        help="Target language (" + "|".join(SUPPORTED_LANGUAGES) + ")",
    )
    parser.add_argument("-m", "--model", dest="model_opt", help="Model path (alt)")
    parser.add_argument("-l", "--lang", dest="lang_opt", help="Target language (alt)")
    parser.add_argument(
        "-o",
        "--output",
        help="Write code to FILE instead of stdout",
    )
    parser.add_argument(
        "--function-name",
        default="predict",
        help="Name of the generated predict function (default: predict)",
    )
    parser.add_argument(
        "--float-type",
        choices=("float", "double"),
        default="float",
        help="Floating-point type for C/C++ (default: float)",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=4,
        help="Spaces per indent level (default: 4); use 0 with --tabs",
    )
    parser.add_argument(
        "--tabs",
        action="store_true",
        help="Indent with tabs instead of spaces",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=6,
        help="Float formatting precision (default: 6)",
    )
    parser.add_argument(
        "--namespace",
        default=None,
        help="Namespace / package for C++, Java, Rust-style backends",
    )
    parser.add_argument(
        "--header",
        action="store_true",
        help="Emit extra includes suitable for C/C++ headers",
    )
    parser.add_argument(
        "--probabilities",
        action="store_true",
        help="Emit class probability distributions instead of hard labels",
    )
    parser.add_argument(
        "--class-names",
        default=None,
        help="Comma-separated class names",
    )
    parser.add_argument(
        "--feature-names",
        default=None,
        help="Comma-separated feature names",
    )
    parser.add_argument(
        "--use-feature-names",
        action="store_true",
        help="Reference features by name when valid identifiers",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Apply IR optimizations (merge identical sibling leaves)",
    )
    parser.add_argument(
        "--no-comments",
        action="store_true",
        help="Omit auto-generated header comments",
    )
    parser.add_argument(
        "--list-languages",
        action="store_true",
        help="Print supported languages and exit",
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def _split_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.list_languages:
        print("\n".join(SUPPORTED_LANGUAGES))
        return 0

    model = args.model_opt or args.model
    language = args.lang_opt or args.language
    if not model or not language:
        parser.error("model and language are required (positional or --model/--lang)")

    options = EmitOptions(
        function_name=args.function_name,
        indent=args.indent if not args.tabs else 1,
        use_tabs=args.tabs,
        float_precision=args.precision,
        float_type=args.float_type,
        class_names=_split_csv(args.class_names),
        feature_names=_split_csv(args.feature_names),
        use_feature_names=args.use_feature_names,
        probabilities=args.probabilities,
        namespace=args.namespace,
        header=args.header,
        include_comments=not args.no_comments,
    )

    try:
        code = convert_file(
            model,
            language,
            options=options,
            feature_names=options.feature_names,
            class_names=options.class_names,
            do_optimize=args.optimize,
        )
    except Mltree2codeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"error: unexpected failure: {exc}", file=sys.stderr)
        return 2

    if args.output:
        Path(args.output).write_text(code, encoding="utf-8")
    else:
        sys.stdout.write(code)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
