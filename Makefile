PYTHON := python

.PHONY: install install-dev format lint test train clean

install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e ".[dev]"

install-dev-wandb:
	$(PYTHON) -m pip install -e ".[dev,wandb]"

format:
	ruff format .
	black .

lint:
	ruff check .

test:
	pytest

train:
	$(PYTHON) train.py

clean:
	rm -rf .pytest_cache .ruff_cache .hydra outputs
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete