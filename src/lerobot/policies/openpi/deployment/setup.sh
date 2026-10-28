#!/usr/bin/env bash
# Isolated OpenPI dependencies; reuse CUDA torch/RDK from the deployment base.
set -euo pipefail
task_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
task_base="${LEROBOT_PYTHON:-/home/nematrix/snap/.conda-envs/lerobot-deploy/bin/python}"
task_uv="${UV_BIN:-$(dirname -- "$task_base")/uv}"
task_env="$task_dir/.venv"
if [[ ! -x "$task_env/bin/python" ]]; then
    "$task_uv" venv --python "$task_base" --system-site-packages "$task_env"
fi
UV_LINK_MODE=copy "$task_uv" pip install --python "$task_env/bin/python" -r "$task_dir/requirements.txt"
"$task_env/bin/python" "$task_dir/../training/install_transformers_patch.py"
printf 'Ready: bash %s/run.sh --check\n' "$task_dir"
