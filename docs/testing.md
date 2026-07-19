# Testing guide

## Layers

1. **Unit** — IR, loader, generators, optimizer (`tests/unit/`).
2. **Integration / semantic** — generated Python `predict` must match
   `model.predict` on random and fixture vectors (`tests/integration/`).
3. **Golden** — formatting snapshots per language (`tests/golden/`).
4. **Performance** — single-sample sklearn vs generated Python if-else
   (`tests/performance/`, marked `@pytest.mark.slow`).
5. **Fixtures** — diverse sklearn trees from `scripts/generate_models.py`.

## Commands

```bash
python scripts/generate_models.py
pytest -q
pytest -q -m slow                  # include performance tests
make bench                         # printable latency comparison
UPDATE_GOLDEN=1 pytest tests/golden -q   # refresh snapshots
```

## Semantic check idea

```python
code = convert(model, "python")
predict = exec_predict(code)
for x in X_test:
    assert predict(x) == model.predict([x])[0]
```

Cross-language compile-and-run checks (GCC, rustc, Node, javac) are optional CI
extensions; the Python path is the primary correctness oracle because all
backends walk the same IR.
