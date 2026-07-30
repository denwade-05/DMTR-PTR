#!/usr/bin/env bash
# Train PolicyNet on synthetic random tensors (short demo).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}/src/policynet"
PYTHON="${PYTHON:-python}"
"${PYTHON}" train.py --dataset srvit_down4_patch4 --exp_name patch4_random --num_iterations 200
