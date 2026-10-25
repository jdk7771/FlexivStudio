#!/usr/bin/env bash
# MANUAL installation only; does not download weights, access hardware, or run deployment.
set -euo pipefail

task_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
task_workspace="$(dirname -- "$task_repo_root")"
task_env_prefix="${LEROBOT_ENV_PREFIX:-$task_workspace/.conda-envs/lerobot-deploy}"
task_conda="${CONDA_EXE:-conda}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$task_workspace/.cache/conda-pkgs}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$task_workspace/.cache/pip}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$task_workspace/.cache/uv}"
export UV_CONCURRENT_DOWNLOADS=16
export UV_HTTP_TIMEOUT=1800
task_pypi="${LEROBOT_PYPI_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"

if [[ -e "$task_env_prefix" ]]; then
    printf '%s\n' "Environment path already exists; choose a new LEROBOT_ENV_PREFIX: $task_env_prefix" >&2
    exit 1
fi
"$task_conda" create --override-channels -c conda-forge --prefix "$task_env_prefix" python=3.12 pip -y
task_python="$task_env_prefix/bin/python"
"$task_python" -m pip install uv --index-url "$task_pypi"
task_uv="$task_env_prefix/bin/uv"
cd "$task_repo_root"
"$task_uv" pip install --python "$task_python" -c examples/flexiv/local-deploy-constraints.txt \
    'torch==2.7.1+cu128' 'torchvision==0.22.1+cu128' \
    --index-url https://download.pytorch.org/whl/cu128 --extra-index-url "$task_pypi" \
    --index-strategy unsafe-best-match
"$task_uv" pip install --python "$task_python" -c examples/flexiv/local-deploy-constraints.txt \
    -e '.[pi,peft]' 'pyrealsense2>=2.55,<3' 'flexivrdk==1.9.2' 'h5py>=3.12,<4' --index-url "$task_pypi"
"$task_conda" env config vars set --prefix "$task_env_prefix" \
    "HF_HOME=$task_workspace/.cache/huggingface" "PIP_CACHE_DIR=$PIP_CACHE_DIR" \
    "UV_CACHE_DIR=$UV_CACHE_DIR" PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
printf '%s\n' "Installed: $task_env_prefix" \
    "Activate with: conda activate $task_env_prefix" \
    'No model was loaded. No robot/camera connection was made.'
