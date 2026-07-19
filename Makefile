.PHONY: install models test lint bench bench-diabetes demo demo-diabetes clean

install:
	pip install -e ".[dev]"

models:
	python scripts/generate_models.py

test: models
	pytest -q

lint:
	ruff check mltree2code tests scripts examples

bench:
	python scripts/benchmark_inference.py

bench-diabetes:
	python scripts/benchmark_diabetes.py

demo:
	python examples/iris_demo.py

demo-diabetes:
	python examples/datacamp_diabetes_demo.py

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
