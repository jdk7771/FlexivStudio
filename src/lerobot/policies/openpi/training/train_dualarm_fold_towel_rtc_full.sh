#!/usr/bin/env bash
set -euo pipefail

task_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
task_root="$(cd -- "$task_dir/.." && pwd)"
task_python="${OPENPI_PYTHON:-/mnt/work/dakun/envs/openpi/bin/python}"
task_config="${OPENPI_CONFIG:-$task_dir/configs/dualarm_fold_towel_rtc_full.json}"
task_print=false
task_check=false
task_resume=""
task_args=()
while (( $# )); do
    case "$1" in
        --print-command) task_print=true; shift ;;
        --check-data) task_check=true; shift ;;
        --resume)
            [[ $# -ge 2 ]] || { echo '--resume requires a checkpoint directory or latest' >&2; exit 2; }
            task_resume="$2"
            shift 2 ;;
        -h|--help)
            echo 'Usage: bash train_dualarm_fold_towel_rtc_full.sh [--print-command | --check-data] [--resume CHECKPOINT_DIR|latest]'
            echo 'Defaults: physical GPU 0,1,2,3; port 29520. CUDA_VISIBLE_DEVICES supports 2 or 4 GPUs; global batch is preserved. OPENPI_CONFIG selects another JSON.'
            exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done
[[ -x "$task_python" ]] || { echo "Run $task_dir/setup.sh first." >&2; exit 1; }
if [[ "$task_resume" == latest ]]; then
    task_resume="$("$task_python" - "$task_config" <<'PY'
import json
from pathlib import Path
import sys

root = Path(json.loads(Path(sys.argv[1]).read_text())["output_dir"]) / "checkpoints"
required = ("train_config.json", "model.safetensors", "training_state.pt")
checkpoints = [p for p in root.iterdir() if p.is_dir() and p.name.isdecimal()
               and all((p / name).is_file() and (p / name).stat().st_size > 0 for name in required)]
if not checkpoints:
    sys.exit(f"No complete checkpoint found in {root}")
print(max(checkpoints, key=lambda p: int(p.name)).resolve())
PY
)"
fi
if [[ -n "$task_resume" ]]; then
    task_resume="$(cd -- "$task_resume" && pwd)"
    for task_file in train_config.json model.safetensors training_state.pt; do
        [[ -s "$task_resume/$task_file" ]] || { echo "Missing checkpoint file: $task_resume/$task_file" >&2; exit 2; }
    done
    task_config="$task_resume/train_config.json"
    task_args+=(--resume "$task_resume")
    echo "Resuming checkpoint: $task_resume" >&2
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export MASTER_PORT="${MASTER_PORT:-29520}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false
export JAX_PLATFORMS=cpu
export HF_HOME="${HF_HOME:-/mnt/work/dakun/data/cache/huggingface}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONPATH="$task_root/src:$task_dir"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
if "$task_check"; then
    export CUDA_VISIBLE_DEVICES=""
    task_command=("$task_python" "$task_dir/train.py" --config "$task_config" --check-data)
else
    IFS=, read -r -a task_devices <<< "$CUDA_VISIBLE_DEVICES"
    [[ ${#task_devices[@]} == 2 || ${#task_devices[@]} == 4 ]] || { echo 'Exactly two or four visible GPUs are required.' >&2; exit 2; }
    task_command=("$task_python" -m torch.distributed.run --nnodes=1 "--nproc-per-node=${#task_devices[@]}"
        "--master-port=$MASTER_PORT" "$task_dir/train.py" --config "$task_config" "${task_args[@]}")
fi
if "$task_print"; then
    printf 'CUDA_VISIBLE_DEVICES=%q JAX_PLATFORMS=cpu ' "$CUDA_VISIBLE_DEVICES"
    printf '%q ' "${task_command[@]}"
    printf '\n'
    exit 0
fi
# The shared base environment provides torchcodec/CUDA/FFmpeg; only the overlay gets patched transformers.
task_base="$($task_python -c 'import sys; print(sys.base_prefix)')"
task_libs="$($task_python - <<'PY'
import pathlib, site
paths = []
for base in site.getsitepackages():
    paths.extend(str(p) for p in pathlib.Path(base).glob('nvidia/*/lib'))
print(':'.join(paths))
PY
)"
export LD_LIBRARY_PATH="$task_base/lib:$task_libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$task_root"
exec "${task_command[@]}"
