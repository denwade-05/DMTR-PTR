#!/usr/bin/env bash
# Evaluate with the synthetic random dataloader (needs a local checkpoint).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}/src"

python main.py \
  --experiment vit_dmtr_ptr_211_k40_t98_aux10_bg40_split5 \
  -m vit_dmtr_ptr \
  --cuda \
  --only-metrics \
  -b 16 \
  --data-name random_down4 \
  --workers 8 \
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
  --ckpt-dir ../ckpt
