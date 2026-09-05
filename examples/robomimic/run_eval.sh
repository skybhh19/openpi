#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <evaluator.py> <train-config> [evaluator arguments...]" >&2
  exit 2
fi

EVALUATOR=$1
TRAIN_CONFIG=$2
shift 2

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

CHECKPOINT_STEP=${CHECKPOINT_STEP:-19999}
PORT=${PORT:-8000}
HOST=${HOST:-localhost}
POLICY_MEM_FRACTION=${POLICY_MEM_FRACTION:-0.8}
POLICY_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
ROBOSUITE_ROOT=${ROBOSUITE_ROOT:-/iris/u/tiangao/projects/robosuite}
SERVER_WAIT_SECONDS=${SERVER_WAIT_SECONDS:-1800}

CHECKPOINT_DIR=${CHECKPOINT_DIR:-checkpoints/$TRAIN_CONFIG/$TRAIN_CONFIG/$CHECKPOINT_STEP}
OUTPUT_PATH=${OUTPUT_PATH:-data/eval_results/${TRAIN_CONFIG}.json}
VIDEO_DIR=${VIDEO_DIR:-data/eval_videos/${TRAIN_CONFIG}}
SERVER_LOG=${SERVER_LOG:-data/eval_logs/${TRAIN_CONFIG}.server.log}

if [[ ! -d "$CHECKPOINT_DIR/params" ]]; then
  echo "Checkpoint params not found: $CHECKPOINT_DIR/params" >&2
  echo "Set CHECKPOINT_STEP or CHECKPOINT_DIR if this run uses a different checkpoint." >&2
  exit 1
fi
if [[ ! -f "$EVALUATOR" ]]; then
  echo "Evaluator not found: $EVALUATOR" >&2
  exit 1
fi
if [[ ! -d "$ROBOSUITE_ROOT" ]]; then
  echo "Robosuite checkout not found: $ROBOSUITE_ROOT" >&2
  echo "Set ROBOSUITE_ROOT to the correct checkout." >&2
  exit 1
fi

mkdir -p "$(dirname -- "$OUTPUT_PATH")" "$VIDEO_DIR" "$(dirname -- "$SERVER_LOG")"

SERVER_PID=
cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "Starting policy server for $TRAIN_CONFIG"
echo "Checkpoint: $CHECKPOINT_DIR"
CUDA_VISIBLE_DEVICES="$POLICY_CUDA_VISIBLE_DEVICES" \
  XLA_PYTHON_CLIENT_MEM_FRACTION="$POLICY_MEM_FRACTION" \
  uv run scripts/serve_policy.py --port="$PORT" policy:checkpoint \
    --policy.config="$TRAIN_CONFIG" \
    --policy.dir="$CHECKPOINT_DIR" \
    >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

echo "Waiting for policy server on $HOST:$PORT (log: $SERVER_LOG)"
server_ready=false
for _ in $(seq 1 $((SERVER_WAIT_SECONDS / 2))); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Policy server exited before becoming ready. Last log lines:" >&2
    tail -50 "$SERVER_LOG" >&2 || true
    exit 1
  fi
  if (exec 3<>"/dev/tcp/$HOST/$PORT") 2>/dev/null; then
    exec 3>&-
    server_ready=true
    break
  fi
  sleep 2
done
if [[ "$server_ready" != true ]]; then
  echo "Timed out after ${SERVER_WAIT_SECONDS}s waiting for policy server. Last log lines:" >&2
  tail -50 "$SERVER_LOG" >&2 || true
  exit 1
fi

echo "Running evaluator: $EVALUATOR"
MUJOCO_GL=${MUJOCO_GL:-egl} \
PYTHONPATH="$REPO_ROOT:$ROBOSUITE_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  uv run "$EVALUATOR" \
    --host="$HOST" \
    --port="$PORT" \
    --output-path="$OUTPUT_PATH" \
    --video-dir="$VIDEO_DIR" \
    "$@"

echo "Evaluation report: $OUTPUT_PATH"
echo "Evaluation videos: $VIDEO_DIR"
