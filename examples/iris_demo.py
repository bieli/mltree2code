from __future__ import annotations

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

from mltree2code import convert
from mltree2code.generators import SUPPORTED_LANGUAGES


iris = load_iris()
model = DecisionTreeClassifier(max_depth=2, random_state=42)
model.fit(iris.data, iris.target)

for lang in SUPPORTED_LANGUAGES:
    print("=" * 60)
    print(lang.upper())
    print("=" * 60)
    print(convert(model, lang))
