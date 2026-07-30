#!/usr/bin/env bash
# Train PolicyNet on synthetic random tensors.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}/src/policynet"

NPROC="${NPROC:-1}"
if [[ "${NPROC}" -gt 1 ]]; then
  torchrun --nproc_per_node="${NPROC}" train.py \
    --dataset srvit_down4_patch4 \
    --exp_name patch4_random \
    --num_iterations 200
else
  python train.py \
    --dataset srvit_down4_patch4 \
    --exp_name patch4_random \
    --num_iterations 200
fi
