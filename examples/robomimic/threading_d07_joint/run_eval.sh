#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DEFAULT_CONFIG=pi05_robomimic_threading_d07_joint_partial_only_low_mem_finetune
TRAIN_CONFIG=${1:-$DEFAULT_CONFIG}
if [[ $# -gt 0 ]]; then
  shift
fi

exec "$SCRIPT_DIR/../run_eval.sh" \
  "$SCRIPT_DIR/main.py" \
  "$TRAIN_CONFIG" \
  "$@"
