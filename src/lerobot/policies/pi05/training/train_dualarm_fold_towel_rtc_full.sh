#!/usr/bin/env bash
# rtc_full: train the entire model, including vision, on four GPUs with RTC; no LoRA.
set -euo pipefail

task_training_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export MASTER_PORT="${MASTER_PORT:-29519}"
exec bash "$task_training_dir/train.sh" "$task_training_dir/configs/dualarm_fold_towel_rtc_full.json" "$@"
