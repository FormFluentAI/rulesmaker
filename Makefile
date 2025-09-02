.PHONY: help venv install setup-cu129 torch-info test clean-venv

PYTHON := .venv/bin/python
PIP := .venv/bin/pip

help:
	@echo "Targets:"
	@echo "  venv         - create local virtualenv in .venv"
	@echo "  install      - install project deps from requirements*.txt"
	@echo "  setup-cu129  - create venv, install deps, torch/cu129"
	@echo "  torch-info   - print torch + CUDA availability"
	@echo "  test         - run pytest in venv"
	@echo "  clean-venv   - remove .venv"

venv:
	python3 -m venv .venv
	$(PYTHON) -m pip install -U pip setuptools wheel

install: venv
	@if [ -f requirements.txt ]; then \
		$(PIP) install -r requirements.txt; \
	fi
	@if [ -f requirements-dev.txt ]; then \
		$(PIP) install -r requirements-dev.txt; \
	fi

setup-cu129: venv
	@echo "Installing project deps and PyTorch (cu129)..."
	$(PYTHON) -m pip install -U pip setuptools wheel
	@if [ -f requirements.txt ]; then \
		$(PIP) install -r requirements.txt; \
	fi
	@if [ -f requirements-dev.txt ]; then \
		$(PIP) install -r requirements-dev.txt; \
	fi
	$(PIP) install torch torchvision --index-url https://download.pytorch.org/whl/cu129

torch-info:
	@$(PYTHON) -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('cuda_version', torch.version.cuda)"

test:
	$(PYTHON) -m pytest -q

clean-venv:
	rm -rf .venv
