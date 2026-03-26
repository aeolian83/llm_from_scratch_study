# LLM Training Lab

Educational repository for building an LLM training stack from scratch.

## Phase 0 goals
- repo structure
- Hydra config system
- Lightning training orchestration
- pure PyTorch model code
- dummy dataset training success
- tooling: Ruff, Black, pytest, pre-commit

## Quick start

```bash
pip install -e ".[dev]"
python train.py
pytest
```