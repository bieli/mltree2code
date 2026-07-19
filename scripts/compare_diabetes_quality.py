"""Compare prediction quality: sklearn diabetes tree vs mltree2code if-else.

Reports how far the transpiled Python if-else path sits from the original
scikit-learn inference in the *value domain*:

1. Hard labels (class 0/1) - agreement rate.
2. Class probabilities - max/mean absolute and L2 distance per sample.
3. Threshold emission error - |emitted - sklearn| for each split at a given
   float precision (shows how rounding can open a disagreement band).

Usage::

    python scripts/compare_diabetes_quality.py
    python scripts/compare_diabetes_quality.py --pruned
    python scripts/compare_diabetes_quality.py --precision 6
    make compare-diabetes
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn import metrics

from mltree2code import convert
from mltree2code.generators.base import EmitOptions
from mltree2code.parser import parse_sklearn_tree
from mltree2code.utils.formatting import format_float

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.datacamp_diabetes_common import (  # noqa: E402
    FEATURE_COLS,
    train_datacamp_diabetes_tree,
)


@dataclass
class QualityReport:
    n_samples: int
    label_matches: int
    label_mismatches: int
    sklearn_accuracy: float
    ifelse_accuracy: float
    # Probability distances (sklearn predict_proba vs generated)
    prob_max_abs: float
    prob_mean_abs: float
    prob_mean_l2: float
    prob_max_l2: float
    # Threshold emission error at used precision
    thr_count: int
    thr_max_abs: float
    thr_mean_abs: float
    thr_rms: float
    precision: int


def _exec_fn(code: str, name: str):
    ns: dict = {}
    exec(code, ns)  # noqa: S102
    return ns[name]


def _threshold_emission_errors(model, precision: int) -> np.ndarray:
    """Absolute error introduced by formatting each split threshold."""
    tree = model.tree_
    errs = []
    for node_id in range(tree.node_count):
        if tree.children_left[node_id] == -1:
            continue
        original = float(tree.threshold[node_id])
        emitted = float(format_float(original, precision=precision))
        errs.append(abs(emitted - original))
    return np.asarray(errs, dtype=np.float64)


def _boundary_margins(model, X: np.ndarray) -> np.ndarray:
    """Min |feature - threshold| along the sklearn path for each sample."""
    tree = model.tree_
    margins = np.empty(len(X), dtype=np.float64)
    for i, x in enumerate(X):
        node = 0
        best = np.inf
        while tree.children_left[node] != -1:
            feat = int(tree.feature[node])
            thr = float(tree.threshold[node])
            best = min(best, abs(float(x[feat]) - thr))
            node = (
                int(tree.children_left[node])
                if float(x[feat]) <= thr
                else int(tree.children_right[node])
            )
        margins[i] = best if best < np.inf else 0.0
    return margins


def compare_quality(
    *,
    pruned: bool = False,
    precision: int = 17,
    split: str = "test",
) -> QualityReport:
    clf, X_train, X_test, y_train, y_test = train_datacamp_diabetes_tree(pruned=pruned)
    if split == "train":
        X, y = X_train, y_train
    elif split == "all":
        X = np.vstack([X_train, X_test])
        y = np.concatenate([y_train, y_test])
    else:
        X, y = X_test, y_test

    opts_hard = EmitOptions(
        function_name="predict_diabetes",
        include_comments=False,
        float_precision=precision,
    )
    opts_prob = EmitOptions(
        function_name="predict_diabetes_proba",
        include_comments=False,
        float_precision=precision,
        probabilities=True,
    )

    code_hard = convert(clf, "python", feature_names=list(FEATURE_COLS), options=opts_hard)
    code_prob = convert(clf, "python", feature_names=list(FEATURE_COLS), options=opts_prob)
    predict = _exec_fn(code_hard, "predict_diabetes")
    predict_proba_gen = _exec_fn(code_prob, "predict_diabetes_proba")

    sk_labels = clf.predict(X).astype(int)
    gen_labels = np.asarray([int(predict(row.tolist())) for row in X], dtype=int)
    sk_proba = clf.predict_proba(X)
    gen_proba = np.asarray(
        [predict_proba_gen(row.tolist()) for row in X],
        dtype=np.float64,
    )

    abs_diff = np.abs(sk_proba - gen_proba)
    l2 = np.linalg.norm(sk_proba - gen_proba, axis=1)
    matches = int(np.sum(sk_labels == gen_labels))
    mismatches = int(len(X) - matches)

    thr_errs = _threshold_emission_errors(clf, precision)

    return QualityReport(
        n_samples=len(X),
        label_matches=matches,
        label_mismatches=mismatches,
        sklearn_accuracy=float(metrics.accuracy_score(y, sk_labels)),
        ifelse_accuracy=float(metrics.accuracy_score(y, gen_labels)),
        prob_max_abs=float(abs_diff.max()) if len(X) else 0.0,
        prob_mean_abs=float(abs_diff.mean()) if len(X) else 0.0,
        prob_mean_l2=float(l2.mean()) if len(X) else 0.0,
        prob_max_l2=float(l2.max()) if len(X) else 0.0,
        thr_count=int(len(thr_errs)),
        thr_max_abs=float(thr_errs.max()) if len(thr_errs) else 0.0,
        thr_mean_abs=float(thr_errs.mean()) if len(thr_errs) else 0.0,
        thr_rms=float(np.sqrt(np.mean(thr_errs**2))) if len(thr_errs) else 0.0,
        precision=precision,
    )


def _print_mismatch_details(pruned: bool, precision: int, split: str, limit: int) -> None:
    clf, X_train, X_test, y_train, y_test = train_datacamp_diabetes_tree(pruned=pruned)
    if split == "train":
        X, y = X_train, y_train
    elif split == "all":
        X = np.vstack([X_train, X_test])
        y = np.concatenate([y_train, y_test])
    else:
        X, y = X_test, y_test

    code = convert(
        clf,
        "python",
        feature_names=list(FEATURE_COLS),
        options=EmitOptions(
            function_name="predict_diabetes",
            include_comments=False,
            float_precision=precision,
        ),
    )
    predict = _exec_fn(code, "predict_diabetes")
    sk = clf.predict(X).astype(int)
    gen = np.asarray([int(predict(row.tolist())) for row in X], dtype=int)
    bad = np.where(sk != gen)[0]
    if len(bad) == 0:
        print("  no label mismatches")
        return

    margins = _boundary_margins(clf, X[bad])
    ir = parse_sklearn_tree(clf, feature_names=list(FEATURE_COLS))
    print(f"  showing up to {min(limit, len(bad))} of {len(bad)} mismatches")
    print(f"  (tree depth={ir.depth()} leaves={ir.leaf_count()})")
    order = np.argsort(margins)  # closest to a split boundary first
    for rank, idx in enumerate(order[:limit]):
        i = int(bad[idx])
        print(
            f"  #{rank + 1}: sample[{i}] true={int(y[i])}  "
            f"sklearn={sk[i]} ifelse={gen[i]}  "
            f"path_margin={margins[idx]:.6g}"
        )
        print(f"       x={np.array2string(X[i], precision=6, separator=', ')}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pruned", action="store_true")
    parser.add_argument(
        "--precision",
        type=int,
        default=17,
        help="Float significant digits used when emitting thresholds (default: 17)",
    )
    parser.add_argument(
        "--split",
        choices=("test", "train", "all"),
        default="test",
        help="Which samples to score (default: test)",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Also compare quality at precisions 6,8,10,12,17",
    )
    parser.add_argument(
        "--show-mismatches",
        type=int,
        default=5,
        help="Print up to N label mismatches with boundary margin (0=skip)",
    )
    args = parser.parse_args(argv)

    mode = "pruned entropy max_depth=3" if args.pruned else "unpruned (DataCamp PNG tree)"
    print("DataCamp diabetes prediction quality")
    print(f"  model:     {mode}")
    print(f"  split:     {args.split}")
    print(f"  precision: {args.precision}")
    print()

    report = compare_quality(
        pruned=args.pruned,
        precision=args.precision,
        split=args.split,
    )

    agree = 100.0 * report.label_matches / report.n_samples
    print("Hard labels (class index)")
    print(f"  samples:            {report.n_samples}")
    print(f"  matches:            {report.label_matches}")
    print(f"  mismatches:         {report.label_mismatches}")
    print(f"  agreement:          {agree:.6f}%")
    print(f"  sklearn vs y_true:  {100 * report.sklearn_accuracy:.4f}% accuracy")
    print(f"  if-else vs y_true:  {100 * report.ifelse_accuracy:.4f}% accuracy")
    print()
    print("Class probabilities (sklearn.predict_proba vs generated return [...])")
    print(f"  max |Δp|:           {report.prob_max_abs:.6e}")
    print(f"  mean |Δp|:          {report.prob_mean_abs:.6e}")
    print(f"  mean L2(Δp):        {report.prob_mean_l2:.6e}")
    print(f"  max  L2(Δp):        {report.prob_max_l2:.6e}")
    print()
    print(f"Threshold emission error at precision={report.precision}")
    print(f"  split thresholds:   {report.thr_count}")
    print(f"  max |thr' - thr|:   {report.thr_max_abs:.6e}")
    print(f"  mean |thr' - thr|:  {report.thr_mean_abs:.6e}")
    print(f"  RMS |thr' - thr|:   {report.thr_rms:.6e}")

    if args.show_mismatches > 0 and report.label_mismatches:
        print()
        print("Mismatch details (smallest path margin first)")
        _print_mismatch_details(args.pruned, args.precision, args.split, args.show_mismatches)

    if args.sweep:
        print()
        print("Precision sweep (test label agreement + max |Δp| + max |Δthr|)")
        print(f"  {'prec':>4}  {'agree%':>10}  {'mism':>5}  {'max|Δp|':>12}  {'max|Δthr|':>12}")
        for prec in (6, 8, 10, 12, 15, 17):
            r = compare_quality(pruned=args.pruned, precision=prec, split=args.split)
            a = 100.0 * r.label_matches / r.n_samples
            print(
                f"  {prec:>4}  {a:10.6f}  {r.label_mismatches:>5}  "
                f"{r.prob_max_abs:12.6e}  {r.thr_max_abs:12.6e}"
            )

    return 0 if report.label_mismatches == 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
