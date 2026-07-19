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

## License

MIT — see [LICENSE](LICENSE).
