# mltree2code

**Machine Learning Tree -> Code** — transpile Machine Learning scikit-learn decision trees into
readable `if`/`else` source for Python, MicroPython, C, C++, Rust, JavaScript, and Java.

Ideal for embedding trained trees on microcontrollers, browsers, or any environment
where you want plain source instead of a runtime ML dependency.

## Install

```bash
pip install -e ".[dev]"
```

## Architecture

```
model (sklearn)  ->  loader  ->  IR  ->  optimizer?  ->  language generator  ->  source
```


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
```

## Development

```bash
make install          # editable install + dev deps
make models           # generate test fixtures
make test             # pytest
make lint             # ruff
UPDATE_GOLDEN=1 make test  # refresh golden snapshots
```

## License

MIT — see [LICENSE](LICENSE).
