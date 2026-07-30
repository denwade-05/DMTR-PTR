#!/usr/bin/env bash
# Smoke test on synthetic random tensors (4x192x384 -> 1x192x384).
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
  --epochs 2 \
  -b 2 \
  --workers 0 \
  --lr 0.0001 \
  --loss genexp \
  --h-patch 4 \
  --h-depth 4 \
  --h-heads 8 \
  --h-dim 64 \
  --h-mlp-dim 128 \
  --h-dim-head 64 \
  --use-early-bypass \
  --super-bypass-layer 3 \
  --enable-base-bypass \
  --base-bypass-layer 4 \
  --base-bg-threshold 0.98 \
  --base-keep-min-ratio 0.4 \
  --bg-aux-weight 0.1 \
  --bg-label-thresh 0.0 \
  --disable-merge \
  --bg-merge-ratio 0.4 \
  --split-ratio 0.05 \
  --dummy-train-samples 8 \
  --dummy-test-samples 4 \
  --ckpt-dir ../ckpt \
  --seed 0 \
  "${EXTRA_CUDA[@]}"
