#!/usr/bin/env bash
# Install an isolated Python environment. This script does not launch training.
set -euo pipefail

task_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../.." && pwd -P)"
task_work_root="$(dirname -- "$task_repo_root")"
if [[ "$(basename -- "$task_work_root")" == repos || "$(basename -- "$task_work_root")" == workspace ]]; then
    task_work_root="$(dirname -- "$task_work_root")"
fi
task_bootstrap="$task_work_root/envs/uv-bootstrap"
task_env="${LEROBOT_ENV:-$task_work_root/envs/lerobot}"
task_system_python="${PYTHON_BOOTSTRAP:-/usr/local/bin/python3}"
if [[ ! -x "$task_system_python" ]]; then
    task_system_python="$(command -v python3)"
fi
export UV_CACHE_DIR="$task_work_root/envs/cache/uv"
export UV_PYTHON_INSTALL_DIR="$task_work_root/envs/python"

if [[ ! -x "$task_bootstrap/bin/uv" ]]; then
    "$task_system_python" -m venv "$task_bootstrap"
    "$task_bootstrap/bin/python" -m pip install uv
fi
task_uv="$task_bootstrap/bin/uv"
"$task_uv" python install 3.12
if [[ ! -x "$task_env/bin/python" ]]; then
    "$task_uv" venv --python 3.12 "$task_env"
fi
task_python="$task_env/bin/python"
cd "$task_repo_root"

# uv pip uses explicit indexes below; it does not apply the project's torch cu128 source mapping.
"$task_uv" pip install --python "$task_python" \
    'torch==2.7.1' 'torchvision==0.22.1' \
    --index-url https://download.pytorch.org/whl/cu118
"$task_uv" pip install --python "$task_python" \
    -c "$task_repo_root/src/lerobot/policies/pi05/training/server-constraints.txt" \
    --index https://download.pytorch.org/whl/cu118 \
    --index-strategy unsafe-best-match \
    -e '.[dataset,training,pi,peft]' h5py pyyaml

"$task_python" - <<'PY'
import sys
import torch
import lerobot
from lerobot.policies.pi05.configuration_pi05 import PI05Config
print('Python:', sys.version.split()[0], 'Torch:', torch.__version__)
print('LeRobot:', lerobot.__file__)
print('RTC configuration available:', PI05Config(rtc_training_max_delay=8).rtc_training_max_delay)
PY
printf '%s\n' 'Environment installed. Training has NOT been started.'
