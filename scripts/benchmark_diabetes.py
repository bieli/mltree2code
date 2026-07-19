#!/usr/bin/env python3
"""Benchmark: sklearn diabetes tree vs mltree2code Python if-else.

Compares single-sample inference latency for the DataCamp Pima Indians
Diabetes decision tree (unpruned by default - same model as the article PNG).

Usage:

    python scripts/benchmark_diabetes.py
    python scripts/benchmark_diabetes.py --pruned
    make bench-diabetes
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from sklearn import metrics

from mltree2code import convert
from mltree2code.generators.base import EmitOptions

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.datacamp_diabetes_common import (  # noqa: E402
    FEATURE_COLS,
    train_datacamp_diabetes_tree,
)


@dataclass
class BenchResult:
    name: str
    n_calls: int
    total_s: float

    @property
    def per_call_us(self) -> float:
        return (self.total_s / self.n_calls) * 1e6

    @property
    def calls_per_s(self) -> float:
        return self.n_calls / self.total_s if self.total_s > 0 else float("inf")


def _exec_predict(code: str, function_name: str = "predict_diabetes"):
    ns: dict = {}
    exec(code, ns)  # noqa: S102
    return ns[function_name]


def _time_calls(fn, samples: list, repeats: int) -> tuple[float, int]:
    for i in range(min(100, len(samples))):
        fn(samples[i % len(samples)])
    start = time.perf_counter()
    n = 0
    for _ in range(repeats):
        for x in samples:
            fn(x)
            n += 1
    return time.perf_counter() - start, n


def run_once(
    *,
    pruned: bool,
    repeats: int,
    max_samples: int,
) -> tuple[BenchResult, BenchResult, float, float]:
    clf, _, X_test, _, y_test = train_datacamp_diabetes_tree(pruned=pruned)
    acc = float(metrics.accuracy_score(y_test, clf.predict(X_test)))

    code = convert(
        clf,
        "python",
        feature_names=list(FEATURE_COLS),
        options=EmitOptions(
            function_name="predict_diabetes",
            include_comments=False,
        ),
    )
    gen_predict = _exec_predict(code)

    n = min(max_samples, len(X_test))
    samples_np = [X_test[i] for i in range(n)]
    samples_list = [x.tolist() for x in samples_np]

    def sklearn_one(x):
        return int(clf.predict([x])[0])

    def generated_one(x):
        return int(gen_predict(x))

    for x, xl in zip(samples_np, samples_list):
        if sklearn_one(x) != generated_one(xl):
            raise AssertionError("generated if-else disagrees with sklearn")

    sk_s, sk_n = _time_calls(sklearn_one, samples_np, repeats)
    gen_s, gen_n = _time_calls(generated_one, samples_list, repeats)
    sk = BenchResult("sklearn DecisionTree.predict([x])", sk_n, sk_s)
    gen = BenchResult("mltree2code Python if-else", gen_n, gen_s)
    speedup = sk.per_call_us / gen.per_call_us if gen.per_call_us > 0 else float("inf")
    return sk, gen, speedup, acc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pruned",
        action="store_true",
        help="Benchmark the pruned entropy depth=3 tree instead of unpruned",
    )
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument(
        "--samples",
        type=int,
        default=231,
        help="Test vectors to score (default: full test set)",
    )
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args(argv)

    mode = "pruned entropy max_depth=3" if args.pruned else "unpruned (DataCamp PNG tree)"
    print("DataCamp diabetes inference benchmark")
    print(f"  model:    {mode}")
    print(f"  features: {list(FEATURE_COLS)}")
    print(f"  samples:  {args.samples}  repeats: {args.repeats}  runs: {args.runs}")
    print()

    speedups: list[float] = []
    last_sk = last_gen = None
    last_acc = 0.0
    for run in range(1, args.runs + 1):
        sk, gen, speedup, acc = run_once(
            pruned=args.pruned,
            repeats=args.repeats,
            max_samples=args.samples,
        )
        last_sk, last_gen, last_acc = sk, gen, acc
        speedups.append(speedup)
        print(f"run {run}/{args.runs}  (test accuracy={acc:.4f})")
        print(
            f"  {sk.name}:  {sk.per_call_us:8.2f} µs/call  "
            f"({sk.calls_per_s:,.0f} calls/s)"
        )
        print(
            f"  {gen.name}:     {gen.per_call_us:8.2f} µs/call  "
            f"({gen.calls_per_s:,.0f} calls/s)"
        )
        print(f"  speedup (sklearn / if-else): {speedup:.2f}x")
        print()

    assert last_sk is not None and last_gen is not None
    med = statistics.median(speedups)
    print("summary (single-sample path, median over runs)")
    print(f"  test accuracy: {last_acc:.6f}")
    print(f"  sklearn:       {last_sk.per_call_us:.2f} µs/call")
    print(f"  if-else:       {last_gen.per_call_us:.2f} µs/call")
    print(f"  speedup:       {med:.2f}x  (generated is faster when > 1)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
