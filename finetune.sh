#!/usr/bin/env bash
set -euo pipefail

# Multi-domain XVLA finetuning launcher for LeRobot-style datasets.
# - Generates X-VLA "general-style" meta JSONs from datasets in /home/nvidia/Documents/datasets
# - Verifies domain handler/ID/weight config exists for each dataset
# - Optionally downloads foundation model from Hugging Face
# - Launches accelerate training

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# --------------------------- User-configurable ---------------------------
MODEL_ID="${MODEL_ID:-2toINF/X-VLA-Pt}"
DATA_ROOT="${DATA_ROOT:-/home/nvidia/Documents/datasets}"
DATASETS_CSV="${DATASETS_CSV:-pickup-blender,pickup-mujoco,insert-blender,insert-mujoco,insert_centrifuge_5430-blender,screw_loose-blender,xarm-lab-data}"

META_DIR="${META_DIR:-${SCRIPT_DIR}/datasets/generated_metas}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs/xvla_multidomain_$(date +%Y%m%d_%H%M%S)}"
PYTHON_BIN="${PYTHON_BIN:-/home/nvidia/miniforge3/envs/XVLA/bin/python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-/home/nvidia/miniforge3/envs/XVLA/bin/accelerate}"

BATCH_SIZE="${BATCH_SIZE:-16}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
VLM_LR_RATIO="${VLM_LR_RATIO:-0.1}"
LEARNING_COEF="${LEARNING_COEF:-${VLM_LR_RATIO}}"
ITERS="${ITERS:-50000}"
FREEZE_STEPS="${FREEZE_STEPS:-0}"
WARMUP_STEPS="${WARMUP_STEPS:-500}"
COSINE_DECAY="${COSINE_DECAY:-1}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
LOG_INTERVAL="${LOG_INTERVAL:-20}"
SEED="${SEED:-0}"
ACTION_MODE="${ACTION_MODE:-auto}"
REAL_ACTION_DIM="${REAL_ACTION_DIM:-14}"
MAX_ACTION_DIM="${MAX_ACTION_DIM:-20}"
NUM_WORKERS="${NUM_WORKERS:-16}"
NUM_PROCESSES="${NUM_PROCESSES:-}"

# Set DOWNLOAD_MODEL=1 to force local snapshot download before training.
DOWNLOAD_MODEL="${DOWNLOAD_MODEL:-0}"
LOCAL_MODEL_DIR="${LOCAL_MODEL_DIR:-${SCRIPT_DIR}/models/X-VLA-Pt}"

# Set PREP_ONLY=1 to only generate metadata/checks and print train command.
PREP_ONLY="${PREP_ONLY:-0}"
# -----------------------------------------------------------------------

echo "== XVLA finetune preflight =="
echo "Model:           ${MODEL_ID}"
echo "Datasets:        ${DATASETS_CSV}"
echo "Data root:       ${DATA_ROOT}"
echo "Generated metas: ${META_DIR}"
echo "Output dir:      ${OUTPUT_DIR}"
echo "Python:          ${PYTHON_BIN}"
echo "Accelerate:      ${ACCELERATE_BIN}"
echo "Base LR:         ${LEARNING_RATE}"
echo "VLM LR ratio:    ${LEARNING_COEF} (expected 0.1 for 1/10)"
echo "VLM freeze:      ${FREEZE_STEPS} steps"
echo "Warmup steps:    ${WARMUP_STEPS}"
echo "Cosine decay:    ${COSINE_DECAY}"
echo "Action mode:     ${ACTION_MODE}"
echo "Real action dim: ${REAL_ACTION_DIM}"
echo "Max action dim:  ${MAX_ACTION_DIM}"
echo "Data workers:    ${NUM_WORKERS}"
echo

mkdir -p "${META_DIR}" "${OUTPUT_DIR}"
export DATA_ROOT META_DIR DATASETS_CSV MODEL_ID LOCAL_MODEL_DIR

if [[ "${LEARNING_COEF}" != "0.1" ]]; then
  echo "WARNING: LEARNING_COEF=${LEARNING_COEF}. For VLM LR = 1/10 base LR, set LEARNING_COEF=0.1"
fi

"${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

from datasets.domain_config import DATA_DOMAIN_ID, DATA_WEIGHTS
from datasets.domain_handler.registry import get_handler_cls

data_root = Path(os.environ.get("DATA_ROOT", "/home/nvidia/Documents/datasets")).resolve()
meta_dir = Path(os.environ.get("META_DIR")).resolve()
datasets_csv = os.environ.get("DATASETS_CSV", "")
datasets = [x.strip() for x in datasets_csv.split(",") if x.strip()]

if not datasets:
    raise SystemExit("No datasets listed in DATASETS_CSV.")

for ds in datasets:
    ds_root = data_root / ds
    info_path = ds_root / "meta" / "info.json"
    episodes_path = ds_root / "meta" / "episodes.jsonl"
    tasks_path = ds_root / "meta" / "tasks.jsonl"

    if not ds_root.exists():
        raise FileNotFoundError(f"Dataset folder not found: {ds_root}")
    if not info_path.exists():
        raise FileNotFoundError(f"Missing info.json: {info_path}")
    if not episodes_path.exists():
        raise FileNotFoundError(f"Missing episodes.jsonl: {episodes_path}")

    # Verify this dataset has a registered handler + domain id/weight.
    get_handler_cls(ds)
    if ds not in DATA_DOMAIN_ID:
        raise KeyError(f"Missing DATA_DOMAIN_ID entry for '{ds}' in datasets/domain_config.py")
    if ds not in DATA_WEIGHTS:
        raise KeyError(f"Missing DATA_WEIGHTS entry for '{ds}' in datasets/domain_config.py")

    with info_path.open("r") as f:
        info = json.load(f)
    datalist = []
    with episodes_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            datalist.append(json.loads(line))

    if not datalist:
        raise ValueError(f"No episodes found in {episodes_path}")

    task_map = {}
    if tasks_path.exists():
        with tasks_path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if "task_index" in row and "task" in row:
                    task_map[int(row["task_index"])] = row["task"]

    meta = {
        "dataset_name": ds,
        "root_path": str(ds_root),
        "data_path": info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"),
        "video_path": info.get("video_path", "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"),
        "chunks_size": int(info.get("chunks_size", 1000)),
        "fps": float(info.get("fps", 30)),
        "task_map": task_map,
        "datalist": datalist,
    }

    out_file = meta_dir / f"{ds}.json"
    with out_file.open("w") as f:
        json.dump(meta, f, indent=2)

    print(f"[meta] wrote {out_file} (episodes={len(datalist)})")

print("[meta] all dataset metas generated successfully.")
PY

if [[ "${DOWNLOAD_MODEL}" == "1" ]]; then
  echo "== Downloading model snapshot =="
  mkdir -p "${LOCAL_MODEL_DIR}"
  if command -v hf >/dev/null 2>&1; then
    hf download "${MODEL_ID}" --local-dir "${LOCAL_MODEL_DIR}"
  elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "${MODEL_ID}" --local-dir "${LOCAL_MODEL_DIR}" --local-dir-use-symlinks False
  else
    "${PYTHON_BIN}" - <<'PY'
from huggingface_hub import snapshot_download
import os
snapshot_download(repo_id=os.environ["MODEL_ID"], local_dir=os.environ["LOCAL_MODEL_DIR"])
PY
  fi
  MODEL_ARG="${LOCAL_MODEL_DIR}"
else
  MODEL_ARG="${MODEL_ID}"
fi

# Set explicit accelerate process count by default to suppress launch warning,
# while preserving existing behavior (same as visible GPU count).
if [[ -z "${NUM_PROCESSES}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_COUNT="$(nvidia-smi -L | wc -l | tr -d ' ')"
    if [[ "${GPU_COUNT}" =~ ^[0-9]+$ ]] && [[ "${GPU_COUNT}" -gt 0 ]]; then
      NUM_PROCESSES="${GPU_COUNT}"
    fi
  fi
fi

TRAIN_CMD=(
  "${ACCELERATE_BIN}" launch
  --num_machines 1
  --dynamo_backend no
  --mixed_precision bf16
  train.py
  --models "${MODEL_ARG}"
  --train_metas_path "${META_DIR}"
  --output_dir "${OUTPUT_DIR}"
  --batch_size "${BATCH_SIZE}"
  --learning_rate "${LEARNING_RATE}"
  --learning_coef "${LEARNING_COEF}"
  --iters "${ITERS}"
  --freeze_steps "${FREEZE_STEPS}"
  --warmup_steps "${WARMUP_STEPS}"
  --save_interval "${SAVE_INTERVAL}"
  --log_interval "${LOG_INTERVAL}"
  --seed "${SEED}"
  --action_mode "${ACTION_MODE}"
  --real_action_dim "${REAL_ACTION_DIM}"
  --max_action_dim "${MAX_ACTION_DIM}"
  --num_workers "${NUM_WORKERS}"
)

if [[ "${COSINE_DECAY}" == "1" ]]; then
  TRAIN_CMD+=( --use_cosine_decay )
fi

if [[ -n "${NUM_PROCESSES}" ]]; then
  TRAIN_CMD=( "${ACCELERATE_BIN}" launch --num_processes "${NUM_PROCESSES}" "${TRAIN_CMD[@]:2}" )
fi

echo
echo "== Launch command =="
printf '%q ' "${TRAIN_CMD[@]}"
echo

if [[ "${PREP_ONLY}" == "1" ]]; then
  echo "PREP_ONLY=1 -> skipping training launch."
  exit 0
fi

"${TRAIN_CMD[@]}"
