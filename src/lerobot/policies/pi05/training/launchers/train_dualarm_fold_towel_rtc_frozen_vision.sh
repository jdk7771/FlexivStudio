#!/usr/bin/env bash
# RTC with a frozen vision encoder; train the remaining model without LoRA.
set -euo pipefail

task_training_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
export MASTER_PORT="${MASTER_PORT:-29517}"
exec bash "$task_training_dir/train.sh" "$task_training_dir/configs/dualarm_fold_towel_rtc_frozen_vision.json" "$@"
