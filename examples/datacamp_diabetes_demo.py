#!/usr/bin/env python3
"""DataCamp Pima diabetes tree -> if-else demo.

Trains the same decision tree as:
https://www.datacamp.com/tutorial/decision-tree-classification-python

By default this is the *unpruned* model whose Graphviz plot is stored at
``examples/data/datacamp_diabetes_tree.png``.

Usage:

    python examples/datacamp_diabetes_demo.py
    python examples/datacamp_diabetes_demo.py --pruned
    python examples/datacamp_diabetes_demo.py --lang c
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sklearn import metrics
from sklearn.tree import export_text

from mltree2code import convert
from mltree2code.generators.base import EmitOptions

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.datacamp_diabetes_common import (  # noqa: E402
    ARTICLE_PRUNED_ACCURACY,
    ARTICLE_UNPRUNED_ACCURACY,
    FEATURE_COLS,
    TREE_PNG,
    train_datacamp_diabetes_tree,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pruned",
        action="store_true",
        help='Use article pruned tree (criterion="entropy", max_depth=3)',
    )
    parser.add_argument(
        "--lang",
        default="python",
        help="Target language for mltree2code (default: python)",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Write generated source to this file (default: stdout)",
    )
    args = parser.parse_args(argv)

    clf, X_train, X_test, y_train, y_test = train_datacamp_diabetes_tree(pruned=args.pruned)
    acc = metrics.accuracy_score(y_test, clf.predict(X_test))
    expected = ARTICLE_PRUNED_ACCURACY if args.pruned else ARTICLE_UNPRUNED_ACCURACY
    mode = "pruned entropy depth=3" if args.pruned else "unpruned (reference PNG)"

    print("DataCamp diabetes decision tree")
    print(f"  mode:           {mode}")
    print(f"  features:       {list(FEATURE_COLS)}")
    print(f"  train/test:     {len(X_train)}/{len(X_test)} (test_size=0.3, random_state=1)")
    print(f"  depth/leaves:   {clf.get_depth()}/{clf.get_n_leaves()}")
    print(f"  test accuracy:  {acc:.6f}  (article reported ~{expected:.6f})")
    if TREE_PNG.is_file():
        print(f"  reference PNG:  {TREE_PNG.relative_to(ROOT)}")
    print()
    print("sklearn export_text (truncated):")
    print(export_text(clf, feature_names=list(FEATURE_COLS), max_depth=4))

    code = convert(
        clf,
        args.lang,
        feature_names=list(FEATURE_COLS),
        class_names=["0", "1"],
        options=EmitOptions(
            function_name="predict_diabetes",
            include_comments=True,
            use_feature_names=False,
        ),
    )
    print(f"=== mltree2code -> {args.lang} ===")
    if args.output:
        Path(args.output).write_text(code, encoding="utf-8")
        print(f"wrote {args.output} ({len(code.splitlines())} lines)")
    else:
        print(code)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
