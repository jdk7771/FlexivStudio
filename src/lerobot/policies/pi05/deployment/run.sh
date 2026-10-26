#!/usr/bin/env bash
# Choose a downloaded RTC-full checkpoint, then use the copied Flexiv deployment.
set -euo pipefail

task_deployment_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
task_repo_root="$(cd -- "$task_deployment_dir/../../../../.." && pwd)"
task_workspace="$(dirname -- "$task_repo_root")"
task_python="${LEROBOT_PYTHON:-}"
if [[ -z "$task_python" ]]; then
    if [[ -n "${VIRTUAL_ENV:-${CONDA_PREFIX:-}}" ]]; then
        task_python="${VIRTUAL_ENV:-$CONDA_PREFIX}/bin/python"
    elif [[ -x "$task_workspace/.conda-envs/lerobot-deploy/bin/python" ]]; then
        task_python="$task_workspace/.conda-envs/lerobot-deploy/bin/python"
    else
        task_python="$(command -v python3)"
    fi
fi
if [[ ! -x "$task_python" ]]; then
    printf '%s\n' "Set LEROBOT_PYTHON to an installed deployment Python: $task_python" >&2
    exit 1
fi
task_checkpoint_args=()
if [[ "${1:-}" =~ ^[0-9]+$ || "${1:-}" == last ]]; then
    task_step="$1"
    case "$task_step" in
        030000|040000|045000|050000|last) ;;
        *) printf '%s\n' 'Choose 030000, 040000, 045000, 050000 or last.' >&2; exit 1 ;;
    esac
    task_checkpoint_args=(--checkpoint "$task_deployment_dir/../checkpoints/rtc_full/$task_step/pretrained_model")
    shift
fi
export TOKENIZERS_PARALLELISM=false
task_site_packages="$("$task_python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
if [[ -d "$task_site_packages/nvidia" ]]; then
    task_nvidia_libs="$(find "$task_site_packages/nvidia" -mindepth 2 -maxdepth 2 -type d -name lib -printf '%p:')"
    export LD_LIBRARY_PATH="${task_nvidia_libs}${LD_LIBRARY_PATH:-}"
fi
task_command=("$task_python" "$task_deployment_dir/deploy_dualarm.py" "${task_checkpoint_args[@]}" "$@")
task_uv="$(dirname -- "$task_python")/uv"
if [[ -x "$task_uv" ]]; then
    task_command=("$task_uv" run --no-project --python "$task_python" python "$task_deployment_dir/deploy_dualarm.py" "${task_checkpoint_args[@]}" "$@")
fi
exec "${task_command[@]}"
