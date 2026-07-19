#!/usr/bin/env python3
"""Compare sklearn DecisionTree inference vs generated Python if-else.

Reports single-sample latency (typical SOC / IoT streaming path) and optional
batch throughput. Run::

    python scripts/benchmark_inference.py
    make bench
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.tree import DecisionTreeClassifier

from mltree2code import convert
from mltree2code.generators.base import EmitOptions


@dataclass
class BenchResult:
    name: str
    n_calls: int
    total_s: float
    predictions_ok: bool

    @property
    def per_call_us(self) -> float:
        return (self.total_s / self.n_calls) * 1e6

    @property
    def calls_per_s(self) -> float:
        return self.n_calls / self.total_s if self.total_s > 0 else float("inf")


def _exec_predict(code: str):
    ns: dict = {}
    exec(code, ns)  # noqa: S102
    return ns["predict"]


def _time_calls(fn, samples: list, repeats: int) -> float:
    # Warm-up
    for _ in range(min(50, len(samples))):
        fn(samples[_ % len(samples)])
    start = time.perf_counter()
    n = 0
    for _ in range(repeats):
        for x in samples:
            fn(x)
            n += 1
    return time.perf_counter() - start, n


def build_model(kind: str, max_depth: int, seed: int = 42):
    if kind == "cancer":
        data = load_breast_cancer()
        X, y = data.data, data.target
    else:
        X, y = make_classification(
            n_samples=2000,
            n_features=20,
            n_informative=12,
            n_redundant=4,
            n_classes=3,
            random_state=seed,
        )
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=seed)
    clf.fit(X, y)
    return clf, X


def run_benchmark(
    *,
    kind: str = "cancer",
    max_depth: int = 8,
    n_samples: int = 200,
    repeats: int = 20,
) -> tuple[BenchResult, BenchResult, float]:
    model, X = build_model(kind, max_depth)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(X), size=min(n_samples, len(X)), replace=False)
    samples = [X[i] for i in idx]
    # list form for generated code (plain Python indexing)
    samples_list = [x.tolist() for x in samples]

    code = convert(model, "python", options=EmitOptions(include_comments=False))
    gen_predict = _exec_predict(code)

    def sklearn_one(x):
        return int(model.predict([x])[0])

    def generated_one(x):
        return int(gen_predict(x))

    # Correctness on the benchmark set
    ok = all(sklearn_one(x) == generated_one(x.tolist()) for x in samples)

    sk_s, sk_n = _time_calls(sklearn_one, samples, repeats)
    gen_s, gen_n = _time_calls(generated_one, samples_list, repeats)

    sk = BenchResult("sklearn DecisionTree.predict([x])", sk_n, sk_s, ok)
    gen = BenchResult("generated Python if-else predict(x)", gen_n, gen_s, ok)
    speedup = sk.per_call_us / gen.per_call_us if gen.per_call_us > 0 else float("inf")
    return sk, gen, speedup


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("cancer", "synthetic"),
        default="cancer",
        help="Dataset used to train the tree (default: cancer)",
    )
    parser.add_argument("--depth", type=int, default=8, help="max_depth (default: 8)")
    parser.add_argument(
        "--samples",
        type=int,
        default=200,
        help="Distinct feature vectors to score each repeat (default: 200)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=25,
        help="Passes over the sample set (default: 25)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Independent timing runs; report median speedup (default: 3)",
    )
    args = parser.parse_args(argv)

    print("mltree2code inference benchmark")
    print(f"  dataset={args.dataset}  max_depth={args.depth}")
    print(f"  samples={args.samples}  repeats={args.repeats}  runs={args.runs}")
    print()

    speedups: list[float] = []
    last_sk = last_gen = None
    for run in range(1, args.runs + 1):
        sk, gen, speedup = run_benchmark(
            kind=args.dataset,
            max_depth=args.depth,
            n_samples=args.samples,
            repeats=args.repeats,
        )
        last_sk, last_gen = sk, gen
        speedups.append(speedup)
        print(f"run {run}/{args.runs}")
        print(
            f"  {sk.name}:  {sk.per_call_us:8.2f} µs/call  "
            f"({sk.calls_per_s:,.0f} calls/s)  n={sk.n_calls}"
        )
        print(
            f"  {gen.name}:  {gen.per_call_us:8.2f} µs/call  "
            f"({gen.calls_per_s:,.0f} calls/s)  n={gen.n_calls}"
        )
        print(f"  speedup (sklearn / generated): {speedup:.2f}x")
        print()

    assert last_sk is not None and last_gen is not None
    if not last_sk.predictions_ok:
        print("ERROR: generated predictions differ from sklearn")
        return 1

    med = statistics.median(speedups)
    print("summary (single-sample path, median over runs)")
    print(f"  sklearn:   {last_sk.per_call_us:.2f} µs/call")
    print(f"  if-else:   {last_gen.per_call_us:.2f} µs/call")
    print(f"  speedup:   {med:.2f}x  (generated is faster when > 1)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
