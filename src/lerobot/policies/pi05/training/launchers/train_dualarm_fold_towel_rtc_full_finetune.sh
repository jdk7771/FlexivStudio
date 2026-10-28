#!/usr/bin/env bash
# Fresh RTC full finetuning on GPUs 6,7; training settings live in the JSON config.
set -euo pipefail

task_training_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
export MASTER_PORT="${MASTER_PORT:-29527}"
exec bash "$task_training_dir/train.sh" "$task_training_dir/configs/dualarm_fold_towel_rtc_full_finetune.json" "$@"
