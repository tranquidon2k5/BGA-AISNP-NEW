#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -d "${PROJECT_ROOT}/.venv" && -f "${PROJECT_ROOT}/.venv/bin/activate" ]]; then
  # Kích hoạt virtualenv nếu có
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/.venv/bin/activate"
fi

python "${PROJECT_ROOT}/scripts/run_all_models.py" "$@"
