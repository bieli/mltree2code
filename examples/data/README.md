# Example data - DataCamp Pima Indians Diabetes

Used by `examples/datacamp_diabetes_demo.py` and `scripts/benchmark_diabetes.py`.

## Files

| File | Description |
|------|-------------|
| `pima-indians-diabetes.csv` | Classic Pima Indians Diabetes dataset (768 rows, no header). Source: [jbrownlee/Datasets](https://github.com/jbrownlee/Datasets). |
| `datacamp_diabetes_tree.png` | Unpruned decision-tree plot from the DataCamp tutorial (Graphviz). |

## Tutorial recipe

From [Decision Tree Classification in Python](https://www.datacamp.com/tutorial/decision-tree-classification-python):

```text
feature_cols = ['pregnant', 'insulin', 'bmi', 'age', 'glucose', 'bp', 'pedigree']
train_test_split(..., test_size=0.3, random_state=1)
DecisionTreeClassifier()                          # unpruned -> reference PNG
DecisionTreeClassifier(criterion="entropy", max_depth=3)  # pruned section
```

CSV columns (headerless): `pregnant,glucose,bp,skin,insulin,bmi,pedigree,age,label`.
