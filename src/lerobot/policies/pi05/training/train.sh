#!/usr/bin/env bash
# Launch one native LeRobot JSON config. --print-command never imports the trainer.
set -euo pipefail

if [[ $# -eq 0 || "${1:-}" == --help || "${1:-}" == -h ]]; then
    printf '%s\n' \
        'Usage: bash train.sh CONFIG.json [--print-command] [LeRobot --key=value ...]' \
        'Resume: bash train.sh CHECKPOINT/pretrained_model/train_config.json --resume=true' \
        'Set CUDA_VISIBLE_DEVICES to choose physical GPUs; LEROBOT_PYTHON selects Python.'
    exit 0
fi

task_training_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
task_repo_root="$(cd -- "$task_training_dir/../../../../.." && pwd -P)"
task_workspace="$(dirname -- "$task_repo_root")"
if [[ "$(basename -- "$task_workspace")" == repos || "$(basename -- "$task_workspace")" == workspace ]]; then
    task_workspace="$(dirname -- "$task_workspace")"
fi
task_python="${LEROBOT_PYTHON:-}"
if [[ -z "$task_python" ]]; then
    if [[ -n "${VIRTUAL_ENV:-${CONDA_PREFIX:-}}" ]]; then
        task_python="${VIRTUAL_ENV:-$CONDA_PREFIX}/bin/python"
    elif [[ -x "$task_repo_root/.venv/bin/python" ]]; then
        task_python="$task_repo_root/.venv/bin/python"
    elif [[ -x "$task_workspace/envs/lerobot/bin/python" ]]; then
        task_python="$task_workspace/envs/lerobot/bin/python"
    else
        task_python="$(command -v python3)"
    fi
fi
if [[ ! -x "$task_python" ]]; then
    printf '%s\n' "Set LEROBOT_PYTHON to a valid Python executable: $task_python" >&2
    exit 1
fi
task_config="$1"
shift
task_print_only=false
task_overrides=()
for task_arg in "$@"; do
    case "$task_arg" in
        --print-command) task_print_only=true ;;
        --config_path|--config_path=*)
            printf '%s\n' 'Pass the config as the first argument, including for resume.' >&2
            exit 1
            ;;
        *) task_overrides+=("$task_arg") ;;
    esac
done

# Only the standard library is used here: command inspection needs no data, weights or CUDA.
task_config_info="$("$task_python" - "$task_config" "${task_overrides[@]}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1]).resolve(strict=True)
if path.suffix != ".json":
    raise SystemExit("Expected a native LeRobot JSON training config.")
config = json.loads(path.read_text())
if config.get("policy", {}).get("type") != "pi05":
    raise SystemExit("This launcher expects policy.type=pi05.")
topology = config.get("parallelism", {})
degrees = {key: topology.get(key, 1) for key in ("dp_replicate", "dp_shard")}
for index, arg in enumerate(sys.argv[2:], 2):
    name, separator, value = arg.partition("=")
    for key in degrees:
        if name == f"--parallelism.{key}":
            degrees[key] = int(value if separator else sys.argv[index + 1])
if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in degrees.values()):
    raise SystemExit("Parallelism degrees must be positive integers.")
print(path)
print(degrees["dp_replicate"] * degrees["dp_shard"])
PY
)"
mapfile -t task_config_fields <<< "$task_config_info"
task_config="${task_config_fields[0]}"
task_workers="${task_config_fields[1]}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export HF_HOME="${HF_HOME:-$task_workspace/datasets/cache/huggingface}"
export PYTHONPATH="$task_repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
task_site_packages="$("$task_python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
if [[ -d "$task_site_packages/nvidia" ]]; then
    task_nvidia_libs="$(find "$task_site_packages/nvidia" -mindepth 2 -maxdepth 2 -type d -name lib -printf '%p:')"
    export LD_LIBRARY_PATH="${task_nvidia_libs}${LD_LIBRARY_PATH:-}"
fi

task_command=("$task_python")
if (( task_workers > 1 )); then
    task_command+=(-m torch.distributed.run "--nproc-per-node=$task_workers" "--master-port=${MASTER_PORT:-29517}")
fi
task_command+=(-m lerobot.scripts.lerobot_train "--config_path=$task_config")
# Machine-specific overrides are explicit. Experiment parameters live in CONFIG.json.
[[ -z "${DATASET_ROOT:-}" ]] || task_command+=("--dataset.root=$DATASET_ROOT")
[[ -z "${MODEL_PATH:-}" ]] || task_command+=("--policy.pretrained_path=$MODEL_PATH")
[[ -z "${TOKENIZER_PATH:-}" ]] || task_command+=("--policy.text_tokenizer_name=$TOKENIZER_PATH")
[[ -z "${OUTPUT_DIR:-}" ]] || task_command+=("--output_dir=$OUTPUT_DIR")
task_command+=("${task_overrides[@]}")
cd "$task_repo_root"
if "$task_print_only"; then
    [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]] || printf 'CUDA_VISIBLE_DEVICES=%q ' "$CUDA_VISIBLE_DEVICES"
    printf '%q ' "${task_command[@]}"
    printf '\n'
    exit 0
fi
exec "${task_command[@]}"
