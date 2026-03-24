#!/usr/bin/env bash
# Download datasets listed in datasets.json from Hugging Face Hub.
#
# Usage:
#   ./download_data.sh                          # download all datasets
#   ./download_data.sh pickup-blender insert-blender  # download specific ones
#
# Environment:
#   DATA_ROOT  – parent directory for downloaded datasets
#                (default: /home/nvidia/Documents/datasets)
#   HF_TOKEN   – optional Hugging Face token for private repos
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASETS_JSON="${SCRIPT_DIR}/datasets.json"
DATA_ROOT="${DATA_ROOT:-/home/nvidia/Documents/datasets}"

if [[ ! -f "${DATASETS_JSON}" ]]; then
  echo "ERROR: datasets.json not found at ${DATASETS_JSON}" >&2
  exit 1
fi

FILTER_ARGS=("$@")

echo "============================================================"
echo "  Dataset Downloader"
echo "  Data root : ${DATA_ROOT}"
echo "  Config    : ${DATASETS_JSON}"
if [[ ${#FILTER_ARGS[@]} -gt 0 ]]; then
  echo "  Filter    : ${FILTER_ARGS[*]}"
fi
echo "============================================================"

python3 - "${DATASETS_JSON}" "${DATA_ROOT}" "${FILTER_ARGS[@]}" <<'PYEOF'
import json
import os
import sys
from pathlib import Path

datasets_json_path = sys.argv[1]
data_root = Path(sys.argv[2])
filters = set(sys.argv[3:])

with open(datasets_json_path) as f:
    all_datasets: dict[str, str] = json.load(f)

if filters:
    datasets = {k: v for k, v in all_datasets.items() if k in filters}
    missing = filters - set(datasets.keys())
    if missing:
        print(f"WARNING: requested datasets not found in datasets.json: {missing}")
else:
    datasets = dict(all_datasets)

if not datasets:
    print("Nothing to download.")
    sys.exit(0)

data_root.mkdir(parents=True, exist_ok=True)

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("ERROR: huggingface_hub is not installed. Run: pip install huggingface_hub", file=sys.stderr)
    sys.exit(1)

for local_name, repo_id in datasets.items():
    dest = data_root / local_name
    print(f"\n{'='*60}")
    print(f"  [{local_name}]  {repo_id}  ->  {dest}")
    print(f"{'='*60}")

    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(dest),
        )
    except Exception as e:
        print(f"  ERROR downloading {repo_id}: {e}")
        continue

    # ------------------------------------------------------------------
    # Post-download: generate compatibility metadata for V3 datasets
    # so that finetune.sh (which expects episodes.jsonl) works.
    # ------------------------------------------------------------------
    info_path = dest / "meta" / "info.json"
    if not info_path.exists():
        print(f"  WARNING: no meta/info.json – skipping metadata generation")
        continue

    with open(info_path) as f:
        info = json.load(f)

    version = info.get("codebase_version", "v2.0")
    print(f"  Format: {version}  |  Episodes: {info.get('total_episodes', '?')}  |  Frames: {info.get('total_frames', '?')}")

    # --- episodes.jsonl ---
    episodes_jsonl = dest / "meta" / "episodes.jsonl"
    episodes_parquet_dir = dest / "meta" / "episodes"

    if not episodes_jsonl.exists() and episodes_parquet_dir.exists():
        print(f"  Generating episodes.jsonl from V3 parquet metadata ...")
        try:
            import pyarrow.parquet as pq
            import glob as _glob

            parquet_files = sorted(_glob.glob(
                str(episodes_parquet_dir / "**" / "*.parquet"), recursive=True
            ))
            episodes = []
            for pf in parquet_files:
                table = pq.read_table(pf)
                for i in range(table.num_rows):
                    row = {col: table.column(col)[i].as_py() for col in table.column_names}
                    episodes.append({
                        "episode_index": row["episode_index"],
                        "tasks": row.get("tasks", []),
                        "length": row.get("length", 0),
                    })

            episodes.sort(key=lambda x: x["episode_index"])
            with open(episodes_jsonl, "w") as f:
                for ep in episodes:
                    f.write(json.dumps(ep) + "\n")
            print(f"  -> wrote {len(episodes)} episodes to episodes.jsonl")
        except Exception as e:
            print(f"  WARNING: failed to generate episodes.jsonl: {e}")

    # --- tasks.jsonl ---
    tasks_jsonl = dest / "meta" / "tasks.jsonl"
    if not tasks_jsonl.exists() and episodes_jsonl.exists():
        print(f"  Generating tasks.jsonl from episodes ...")
        try:
            task_set: dict[str, int] = {}
            with open(episodes_jsonl) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    ep = json.loads(line)
                    for task_str in ep.get("tasks", []):
                        if task_str not in task_set:
                            task_set[task_str] = len(task_set)

            with open(tasks_jsonl, "w") as f:
                for task_str, task_idx in task_set.items():
                    f.write(json.dumps({"task_index": task_idx, "task": task_str}) + "\n")
            print(f"  -> wrote {len(task_set)} tasks to tasks.jsonl")
        except Exception as e:
            print(f"  WARNING: failed to generate tasks.jsonl: {e}")

    # --- Ensure chunks_size is present in info.json (some V3 exports omit it) ---
    if "chunks_size" not in info and version.startswith("v3"):
        info["chunks_size"] = 1000
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)
            f.write("\n")
        print(f"  -> added default chunks_size=1000 to info.json")

    print(f"  Done: {local_name}")

print(f"\n{'='*60}")
print(f"All downloads complete.  DATA_ROOT={data_root}")
print(f"{'='*60}")
PYEOF
