# mltree2code

**Machine Learning Tree -> Code** — transpile Machine Learning scikit-learn decision trees into
readable `if`/`else` source for Python, MicroPython, C, C++, Rust, JavaScript, and Java.

Ideal for embedding trained trees on microcontrollers, browsers, or any environment
where you want plain source instead of a runtime ML dependency.

## Motivators

Many **SOC** and **IoT / IIoT** pipelines need decision-tree decisions in the hot path —
packet/feature triage, anomaly flags, thresholded sensor rules — but cannot afford a full
ML runtime on every hop or device.

Typical constraints that push teams toward generated `if`/`else` trees:

- **Latency** — native C/C++/Rust (or MicroPython on MCUs) avoids Python/sklearn call
  overhead and interpreter cost when scoring thousands of events per second.
- **Footprint** — edge gateways, PLCs, and constrained sensors often have no room for
  NumPy/scikit-learn; a few dozen lines of compiled conditions fit in flash/RAM.
- **Determinism & audit** — plain branches are easy to review, unit-test, and ship under
  change-control (useful for SOC playbooks and industrial safety/compliance reviews).
- **Offline / air-gapped** — inference must keep working without model servers or cloud
  round-trips; the tree ships as firmware or a small library.
- **Same model, many targets** — train once in sklearn, emit C for a sensor node, Rust for
  a gateway, and JavaScript for an operator console without retraining.

`mltree2code` turns that workflow into a one-step transpile: train offline, generate
source, compile or embed, and keep inference fast where it actually runs.

## Install

```bash
pip install -e ".[dev]"
```

## Quick start

```bash
# Train & save a tree, then:
mltree2code model.joblib python
mltree2code model.joblib c > tree.c
mltree2code model.joblib rust --function-name score
mltree2code --model tree.pkl --lang javascript -o tree.js
```

### Python API

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import joblib

from mltree2code import convert, convert_file

iris = load_iris()
clf = DecisionTreeClassifier(max_depth=2, random_state=42)
clf.fit(iris.data, iris.target)
joblib.dump(clf, "iris.joblib")

print(convert(clf, "python"))
print(convert_file("iris.joblib", "c"))
```

### Example output (Python)

```python
def predict(x):
    if x[2] <= 2.45:
        return 0
    else:
        if x[3] <= 1.75:
            return 1
        else:
            return 2
```

## Architecture

```
model (sklearn)  ->  loader  ->  Intermediate Representation (IR)  ->  optimizer?  ->  language generator  ->  source
```

Adding a new language only requires a new generator under `mltree2code/generators/`.

| Layer | Role |
|-------|------|
| `loader.py` | Load pickle/joblib, validate estimator type |
| `parser.py` | Walk `model.tree_` -> `TreeIR` |
| `ir.py` | Language-independent `Node` / `Leaf` |
| `optimizer.py` | Optional IR rewrites (merge identical leaves) |
| `generators/` | Emit target-language syntax |
| `cli.py` | argparse front-end |

## Supported models (v0.1)

- `sklearn.tree.DecisionTreeClassifier`
- `sklearn.tree.DecisionTreeRegressor`

**Planned:** RandomForest, ExtraTrees, XGBoost, LightGBM, CatBoost.

## Supported languages

| Language | CLI name | Aliases |
|----------|----------|---------|
| Python | `python` | `py` |
| MicroPython | `micropython` | `upy` |
| C | `c` | |
| C++ | `cpp` | `c++`, `cxx` |
| Rust | `rust` | `rs` |
| JavaScript | `javascript` | `js`, `node` |
| Java | `java` | |

## CLI reference

```
mltree2code MODEL LANGUAGE [options]

  -o, --output FILE          Write to file instead of stdout
  --function-name NAME       Predict function name (default: predict)
  --float-type float|double  C/C++ float type
  --indent N                 Spaces per indent (default: 4)
  --tabs                     Use tab indentation
  --precision N              Float format precision
  --namespace NAME           C++/Java package or namespace
  --header                   Emit C/C++ includes
  --probabilities            Emit class probabilities
  --class-names a,b,c        Class labels
  --feature-names ...        Feature labels
  --use-feature-names        Prefer feature names in conditions
  --optimize                 Merge identical sibling leaves
  --no-comments              Skip header comments
  --list-languages           Print backends and exit
```

## Development

```bash
make install          # editable install + dev deps
make models           # generate test fixtures
make test             # pytest
make lint             # ruff
UPDATE_GOLDEN=1 make test  # refresh golden snapshots
```

See [docs/](docs/) for Intermediate Representation (IR) design, extending generators, and the testing guide.

## License

MIT — see [LICENSE](LICENSE).
