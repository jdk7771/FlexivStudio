#!/usr/bin/env bash
set -euo pipefail

task_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
task_root="$(cd -- "$task_dir/.." && pwd)"
task_python="${OPENPI_PYTHON:-/mnt/work/dakun/envs/openpi/bin/python}"
task_config="${OPENPI_CONFIG:-$task_dir/configs/dualarm_fold_towel_pose_delta_no_rtc_full.json}"
task_mode=train
task_print=false
task_resume=""
task_args=()
while (( $# )); do
    case "$1" in
        --print-command) task_print=true; shift ;;
        --prepare-data|--check-data|--test)
            [[ "$task_mode" == train ]] || { echo 'Select one of --prepare-data, --check-data, --test.' >&2; exit 2; }
            task_mode="${1#--}"; shift ;;
        --resume)
            [[ $# -ge 2 && "$2" != --* ]] || { echo '--resume requires a pose checkpoint directory or latest.' >&2; exit 2; }
            task_resume="$2"; shift 2 ;;
        -h|--help)
            echo 'Usage: bash train_dualarm_fold_towel_pose_delta_no_rtc_full.sh [--prepare-data|--check-data|--test] [--print-command] [--resume DIR|latest]'
            echo 'Official pi0.5 pose-delta run without RTC. Defaults: GPU 0,1; batch 16/rank; port 29526. OPENPI_CONFIG and OPENPI_PYTHON may be overridden.'
            exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done
[[ -x "$task_python" ]] || { echo "Missing existing OpenPI environment: $task_python" >&2; exit 1; }
if [[ -n "$task_resume" ]]; then
    [[ "$task_mode" == train || "$task_mode" == check-data ]] || { echo 'Resume only applies to training/check-data.' >&2; exit 2; }
    task_resume="$("$task_python" - "$task_config" "$task_resume" <<'PY'
import json
from pathlib import Path
import sys

required = ('model.safetensors', 'training_state.pt', 'train_config.json', 'pose_delta_stats.json')
def complete(path):
    return path.is_dir() and all((path / name).is_file() and (path / name).stat().st_size for name in required)
if sys.argv[2] == 'latest':
    root = Path(json.loads(Path(sys.argv[1]).read_text())['output_dir']) / 'checkpoints'
    candidates = [p for p in root.iterdir() if p.name.isdecimal() and complete(p)] if root.is_dir() else []
    if not candidates:
        sys.exit(f'No complete pose checkpoint in {root}')
    path = max(candidates, key=lambda p: int(p.name))
else:
    path = Path(sys.argv[2])
if not complete(path):
    sys.exit(f'Incomplete pose checkpoint: {path}; joint checkpoints are incompatible.')
if 'pose_delta' not in json.loads((path / 'train_config.json').read_text()):
    sys.exit('Checkpoint is not a pose-delta run.')
print(path.resolve())
PY
)"
    task_config="$task_resume/train_config.json"
    task_args+=(--resume "$task_resume")
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export MASTER_PORT="${MASTER_PORT:-29526}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false
export JAX_PLATFORMS=cpu
export HF_HOME="${HF_HOME:-/mnt/work/dakun/data/cache/huggingface}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$task_root/src:$task_dir"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
case "$task_mode" in
    prepare-data|check-data|test)
        export CUDA_VISIBLE_DEVICES=""
        export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
        if [[ "$task_mode" == prepare-data ]]; then
            task_command=("$task_python" "$task_dir/prepare_dataset.py" --config "$task_config")
        elif [[ "$task_mode" == check-data ]]; then
            task_command=("$task_python" "$task_dir/train.py" --config "$task_config" --check-data)
        else
            task_command=("$task_python" -m pytest -q -c /dev/null --confcutdir="$task_dir" -p no:cacheprovider "$task_dir/test_pose_delta.py")
        fi ;;
    train)
        IFS=, read -r -a task_devices <<< "$CUDA_VISIBLE_DEVICES"
        [[ ${#task_devices[@]} == 2 || ${#task_devices[@]} == 4 ]] || { echo 'Exactly two or four visible GPUs are required.' >&2; exit 2; }
        "$task_python" - "$task_config" "${#task_devices[@]}" <<'PY'
import json
from pathlib import Path
import sys

config = json.loads(Path(sys.argv[1]).read_text())
expected = {"dp_replicate": int(sys.argv[2]), "dp_shard": 1}
if config["parallelism"] != expected:
    sys.exit(f'Visible GPU count must match config parallelism: {config["parallelism"]}.')
PY
        task_command=("$task_python" -m torch.distributed.run --nnodes=1 "--nproc-per-node=${#task_devices[@]}"
            "--master-port=$MASTER_PORT" "$task_dir/train.py" --config "$task_config" "${task_args[@]}") ;;
esac
if "$task_print"; then
    printf 'CUDA_VISIBLE_DEVICES=%q JAX_PLATFORMS=cpu ' "$CUDA_VISIBLE_DEVICES"
    printf '%q ' "${task_command[@]}"
    printf '\n'
    exit 0
fi
task_base="$("$task_python" -c 'import sys; print(sys.base_prefix)')"
task_libs="$("$task_python" - <<'PY'
import pathlib
import site
print(':'.join(str(p) for base in site.getsitepackages() for p in pathlib.Path(base).glob('nvidia/*/lib')))
PY
)"
export LD_LIBRARY_PATH="$task_base/lib:$task_libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$task_root"
exec "${task_command[@]}"
