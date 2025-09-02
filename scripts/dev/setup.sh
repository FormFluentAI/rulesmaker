#!/usr/bin/env bash
set -euo pipefail

CU_INDEX_URL="https://download.pytorch.org/whl/cu129"

if [[ ! -d .venv ]]; then
  echo "[setup] Creating virtualenv at .venv"
  python3 -m venv .venv
fi

echo "[setup] Upgrading pip/setuptools/wheel"
. .venv/bin/activate
python -m pip install -U pip setuptools wheel

if [[ -f requirements.txt ]]; then
  echo "[setup] Installing requirements.txt"
  pip install -r requirements.txt
fi

if [[ -f requirements-dev.txt ]]; then
  echo "[setup] Installing requirements-dev.txt"
  pip install -r requirements-dev.txt
fi

echo "[setup] Installing torch/torchvision from cu129 channel"
pip install torch torchvision --index-url "$CU_INDEX_URL"

echo "[setup] Verifying CUDA availability"
python - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
print('cuda_version', torch.version.cuda)
PY

echo "[setup] Done. Activate with: source .venv/bin/activate"

