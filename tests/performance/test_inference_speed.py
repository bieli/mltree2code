"""Performance: sklearn tree vs generated Python if-else inference."""

from __future__ import annotations

import time

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier

from mltree2code import convert
from mltree2code.generators.base import EmitOptions


def _exec_predict(code: str):
    ns: dict = {}
    exec(code, ns)  # noqa: S102
    return ns["predict"]


def _median_us(fn, samples, repeats: int = 5) -> float:
    times: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        for x in samples:
            fn(x)
        times.append((time.perf_counter() - start) / len(samples) * 1e6)
    times.sort()
    return times[len(times) // 2]


@pytest.mark.slow
def test_generated_python_faster_than_sklearn_single_sample():
    """Single-sample path: generated if-else should beat sklearn.predict([x])."""
    data = load_breast_cancer()
    model = DecisionTreeClassifier(max_depth=8, random_state=42)
    model.fit(data.data, data.target)

    rng = np.random.default_rng(0)
    idx = rng.choice(len(data.data), size=150, replace=False)
    samples_np = [data.data[i] for i in idx]
    samples_list = [x.tolist() for x in samples_np]

    code = convert(model, "python", options=EmitOptions(include_comments=False))
    gen_predict = _exec_predict(code)

    def sklearn_one(x):
        return int(model.predict([x])[0])

    def generated_one(x):
        return int(gen_predict(x))

    # Semantic check on the timed set
    for x, xl in zip(samples_np, samples_list):
        assert sklearn_one(x) == generated_one(xl)

    # Warm-up
    for x, xl in zip(samples_np[:20], samples_list[:20]):
        sklearn_one(x)
        generated_one(xl)

    sk_us = _median_us(sklearn_one, samples_np, repeats=5)
    gen_us = _median_us(generated_one, samples_list, repeats=5)
    speedup = sk_us / gen_us

    # Require a clear win on the streaming single-sample path.
    assert gen_us < sk_us, (
        f"expected generated if-else faster than sklearn single-sample; "
        f"sklearn={sk_us:.2f}µs generated={gen_us:.2f}µs"
    )
    assert speedup >= 1.5, (
        f"expected >= 1.5x speedup on single-sample path; got {speedup:.2f}x "
        f"(sklearn={sk_us:.2f}µs, generated={gen_us:.2f}µs)"
    )


@pytest.mark.slow
def test_benchmark_script_runs():
    """Smoke: CLI benchmark exits 0 and prints a speedup line."""
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    proc = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "benchmark_inference.py"),
            "--dataset",
            "cancer",
            "--depth",
            "5",
            "--samples",
            "80",
            "--repeats",
            "3",
            "--runs",
            "1",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=root,
    )
    assert proc.returncode == 0, proc.stderr
    assert "sklearn" in proc.stdout
    assert "if-else" in proc.stdout
    assert "speedup" in proc.stdout
