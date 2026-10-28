#!/usr/bin/env bash
# Usage: bash run.sh [<experiment>] [<step>] [--no-rtc] [deployment options]
set -euo pipefail
task_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
task_python="${OPENPI_DEPLOY_PYTHON:-$task_dir/.venv/bin/python}"
if [[ ! -x "$task_python" ]]; then
    printf 'Create the isolated environment first: bash %s/setup.sh\n' "$task_dir" >&2
    exit 1
fi
task_args=()
if [[ -n "${1:-}" && "$1" != -* ]]; then
    task_group=openpi_pi05_dualarm_fold_towel_rtc_full
    if [[ ! "$1" =~ ^[0-9]+$ ]]; then
        task_group="$1"
        if [[ "$task_group" == .* || "$task_group" == */* ]]; then
            printf 'Invalid experiment folder: %s\n' "$task_group" >&2
            exit 1
        fi
        shift
    fi
    if [[ ! "${1:-}" =~ ^[0-9]+$ ]]; then
        printf 'Usage: bash run.sh [<experiment>] <step> [options]\n' >&2
        exit 1
    fi
    task_step="$(printf '%06d' "$((10#$1))")"
    task_args=(--checkpoint "$task_dir/../checkpoints/$task_group/$task_step")
    shift
fi
export JAX_PLATFORMS=cpu HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
# CUDA libraries inherited from the existing deployment environment.
task_cuda_paths="$("$task_python" -c 'import site; from pathlib import Path; print(":".join(str(p) for s in site.getsitepackages() for p in Path(s).glob("nvidia/*/lib")))')"
export LD_LIBRARY_PATH="${task_cuda_paths}:${LD_LIBRARY_PATH:-}"
exec "$task_python" "$task_dir/deploy.py" "${task_args[@]}" "$@"
