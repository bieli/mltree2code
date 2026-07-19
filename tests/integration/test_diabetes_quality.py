from __future__ import annotations

import pytest

from scripts.compare_diabetes_quality import compare_quality


@pytest.mark.parametrize("pruned", [False, True])
def test_diabetes_quality_perfect_at_default_precision(pruned):
    report = compare_quality(pruned=pruned, precision=17, split="test")
    assert report.label_mismatches == 0
    assert report.prob_max_abs == 0.0
    assert report.thr_max_abs == 0.0


def test_diabetes_quality_low_precision_can_diverge():
    # Unpruned tree: coarse threshold rounding can flip a boundary sample.
    low = compare_quality(pruned=False, precision=6, split="test")
    high = compare_quality(pruned=False, precision=17, split="test")
    assert high.label_mismatches == 0
    assert low.thr_max_abs > high.thr_max_abs
    # At least one of label or probability distance should show the gap.
    assert low.label_mismatches >= 1 or low.prob_max_abs > 0
