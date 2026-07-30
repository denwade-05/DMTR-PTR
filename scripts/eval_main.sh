#!/usr/bin/env bash
# Evaluation needs a local checkpoint; use after smoke/train has produced one.
# Kept minimal on purpose — no paper hyper-parameter dump here.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}/src"
PYTHON="${PYTHON:-python}"
EXTRA_CUDA=()
if "${PYTHON}" -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)'; then
  EXTRA_CUDA=(--cuda --gpu 0)
fi
"${PYTHON}" main.py \
  --experiment smoke_random_dmtr_ptr \
  -m vit_dmtr_ptr \
  --only-metrics \
  -b 2 \
  --workers 0 \
  --loss genexp \
  --disable-merge \
  --dummy-test-samples 4 \
  --ckpt-dir ../ckpt \
  "${EXTRA_CUDA[@]}"
