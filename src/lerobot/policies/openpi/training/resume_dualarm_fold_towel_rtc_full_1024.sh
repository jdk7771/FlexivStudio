#!/usr/bin/env bash
# Resume the shared OpenPI full fine-tuning run on server 1024, physical GPUs 0-3.
set -euo pipefail

task_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_PORT="${MASTER_PORT:-29524}"
exec bash "$task_dir/train_dualarm_fold_towel_rtc_full.sh" --resume latest "$@"
