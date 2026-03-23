#!/usr/bin/env bash
set -euo pipefail

# Stage-2 finetuning wrapper:
# - resume from stage-1 checkpoint
# - LR reduced by 10x
# - xarm sampling weight multiplied by 2x
# - train for +20k steps
#
# Inherits ACTION_MODE from finetune.sh (default: joint).
# Override with e.g. ACTION_MODE=auto to change.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Stage-1 checkpoint to continue from (override if needed)
MODEL_ID="${MODEL_ID:-/home/nvidia/Documents/X-VLA/outputs/xvla_multidomain_20260223_085934/ckpt-50000}"

# Stage-2 defaults
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
ITERS="${ITERS:-20000}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs/xvla_multidomain_stg2_$(date +%Y%m%d_%H%M%S)}"

# Increase xarm sampling importance by 2x at runtime.
XARM_WEIGHT_MULTIPLIER="${XARM_WEIGHT_MULTIPLIER:-2.0}"

echo "== Stage-2 XVLA finetune =="
echo "Resume model:            ${MODEL_ID}"
echo "Stage-2 LR:              ${LEARNING_RATE}"
echo "Stage-2 steps:           ${ITERS}"
echo "Xarm weight multiplier:  ${XARM_WEIGHT_MULTIPLIER}"
echo "Output dir:              ${OUTPUT_DIR}"
echo

export MODEL_ID
export LEARNING_RATE
export ITERS
export OUTPUT_DIR
export XARM_WEIGHT_MULTIPLIER

# Delegate to the main launcher so all existing settings remain consistent.
bash "${SCRIPT_DIR}/finetune.sh"

