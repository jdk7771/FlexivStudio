#!/usr/bin/env bash
# CPU-only setup for this machine. Never installs into the existing LeRobot environment.
set -euo pipefail
task_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
task_root="$(cd -- "$task_dir/.." && pwd)"
task_base="${LEROBOT_PYTHON:-/mnt/work/dakun/conda-envs/lerobot/bin/python}"
if [[ ! -x "$task_root/.venv/bin/python" ]]; then
    uv venv --python "$task_base" --system-site-packages "$task_root/.venv"
fi
UV_LINK_MODE=copy uv pip install --index-url "${OPENPI_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple}" \
    --python "$task_root/.venv/bin/python" -r "$task_dir/requirements.txt"
UV_LINK_MODE=copy uv pip install --index-url "${OPENPI_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple}" \
    --python "$task_root/.venv/bin/python" --no-deps 'accelerate==1.10.1' 'lerobot==0.4.4'
"$task_root/.venv/bin/python" "$task_dir/install_transformers_patch.py"
echo "Ready: bash $task_dir/train_dualarm_fold_towel_rtc_full.sh --check-data"
