# Intermediate Representation (IR)

`mltree2code` never generates language syntax directly from sklearn arrays.
Models are first lowered to a small, language-independent tree:

```python
Node(feature=2, threshold=1.75, left=..., right=...)
Leaf(prediction=1, probabilities=[0.0, 1.0, 0.0])
```

## Why IR?

- One parser for sklearn; N backends that only care about syntax.
- Optimizations (leaf merging, future constant folding) run once on IR.
- Golden / semantic tests can assert on structure independently of formatting.

## Types

| Type | Meaning |
|------|---------|
| `Leaf` | Terminal prediction (class index or float) |
| `Node` | Split: `x[feature] <= threshold` -> left, else right |
| `TreeIR` | Root + metadata (`n_features`, `n_classes`, names, task) |

Sklearn uses the same left/right convention (`children_left` / `children_right`
with `-1` for leaves). The parser preserves that semantics exactly so generated
code matches `model.predict`.
