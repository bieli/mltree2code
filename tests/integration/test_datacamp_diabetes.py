from __future__ import annotations

import numpy as np
import pytest
from sklearn import metrics

from examples.datacamp_diabetes_common import (
    ARTICLE_PRUNED_ACCURACY,
    ARTICLE_UNPRUNED_ACCURACY,
    CSV_PATH,
    FEATURE_COLS,
    TREE_PNG,
    train_datacamp_diabetes_tree,
)
from mltree2code import convert
from mltree2code.generators.base import EmitOptions


def _exec_predict(code: str, name: str = "predict_diabetes"):
    ns: dict = {}
    exec(code, ns)  # noqa: S102
    return ns[name]


@pytest.fixture(scope="module")
def unpruned_diabetes():
    return train_datacamp_diabetes_tree(pruned=False)


@pytest.fixture(scope="module")
def pruned_diabetes():
    return train_datacamp_diabetes_tree(pruned=True)


def test_datacamp_data_files_exist():
    assert CSV_PATH.is_file()
    assert TREE_PNG.is_file()


def test_unpruned_matches_article_accuracy(unpruned_diabetes):
    clf, _, X_test, _, y_test = unpruned_diabetes
    acc = metrics.accuracy_score(y_test, clf.predict(X_test))
    assert acc == pytest.approx(ARTICLE_UNPRUNED_ACCURACY)
    assert clf.get_depth() >= 10
    assert clf.get_n_leaves() >= 50


def test_pruned_matches_article_accuracy(pruned_diabetes):
    clf, _, X_test, _, y_test = pruned_diabetes
    acc = metrics.accuracy_score(y_test, clf.predict(X_test))
    assert acc == pytest.approx(ARTICLE_PRUNED_ACCURACY)
    assert clf.get_depth() == 3


def test_unpruned_root_is_glucose(unpruned_diabetes):
    clf, *_ = unpruned_diabetes
    root_feat = int(clf.tree_.feature[0])
    assert FEATURE_COLS[root_feat] == "glucose"


def test_unpruned_python_ifelse_matches_sklearn(unpruned_diabetes):
    clf, _, X_test, _, _ = unpruned_diabetes
    code = convert(
        clf,
        "python",
        feature_names=list(FEATURE_COLS),
        options=EmitOptions(function_name="predict_diabetes", include_comments=False),
    )
    predict = _exec_predict(code)
    assert "if " in code
    assert code.count("if ") >= 20  # deep unpruned tree
    for x in X_test:
        assert int(predict(x.tolist())) == int(clf.predict([x])[0])


def test_pruned_python_ifelse_matches_sklearn(pruned_diabetes):
    clf, _, X_test, _, _ = pruned_diabetes
    code = convert(
        clf,
        "python",
        feature_names=list(FEATURE_COLS),
        options=EmitOptions(function_name="predict_diabetes", include_comments=False),
    )
    predict = _exec_predict(code)
    for x in X_test:
        assert int(predict(x.tolist())) == int(clf.predict([x])[0])


@pytest.mark.slow
def test_unpruned_ifelse_faster_than_sklearn_single_sample(unpruned_diabetes):
    import time

    clf, _, X_test, _, _ = unpruned_diabetes
    code = convert(
        clf,
        "python",
        options=EmitOptions(function_name="predict_diabetes", include_comments=False),
    )
    predict = _exec_predict(code)
    samples = [X_test[i] for i in range(min(100, len(X_test)))]
    lists = [s.tolist() for s in samples]

    def sk(x):
        return int(clf.predict([x])[0])

    def gen(x):
        return int(predict(x))

    for _ in range(20):
        sk(samples[0])
        gen(lists[0])

    def median_us(fn, xs, repeats=5):
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            for x in xs:
                fn(x)
            times.append((time.perf_counter() - t0) / len(xs) * 1e6)
        return float(np.median(times))

    sk_us = median_us(sk, samples)
    gen_us = median_us(gen, lists)
    assert gen_us < sk_us
    assert sk_us / gen_us >= 1.5
