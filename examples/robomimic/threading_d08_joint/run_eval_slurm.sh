#!/usr/bin/env bash
#SBATCH --partition=iris-hi
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:l40s:1
#SBATCH --constraint=ada
#SBATCH --account=iris
#SBATCH --output=data/eval_logs/%x-%j.slurm.log

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: sbatch $0 <train-config> [evaluator arguments...]" >&2
  exit 2
fi

REPO_ROOT=${SLURM_SUBMIT_DIR:-/iris/u/tiangao/projects/openpi}
SCRIPT_DIR="$REPO_ROOT/examples/robomimic/threading_d08_joint"
cd "$REPO_ROOT"

export HF_LEROBOT_HOME=${HF_LEROBOT_HOME:-/iris/u/tiangao/lerobot_datasets}
export PORT=${PORT:-$((10000 + SLURM_JOB_ID % 50000))}

exec "$SCRIPT_DIR/run_eval.sh" "$@"
