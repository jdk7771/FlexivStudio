#!/usr/bin/env bash
# Shortcut to an experiment config; training parameters are stored in JSON.
set -euo pipefail

task_training_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
exec bash "$task_training_dir/train.sh" "$task_training_dir/configs/dualarm_fold_towel_rtc_lora.json" "$@"
