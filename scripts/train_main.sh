#!/usr/bin/env bash
# Same entry as the smoke demo (synthetic data).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "${ROOT}/scripts/smoke_random.sh"
