#!/usr/bin/env bash
# DualArm_FoldTowel_pi05_no_rtc_full: full fine-tuning, default GPUs 3,2,1,0, global batch 32.
set -euo pipefail

task_training_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3,2,1,0}"
export MASTER_PORT="${MASTER_PORT:-29518}"
task_overrides=()
[[ -z "${BATCH_SIZE:-}" ]] || task_overrides+=("--batch_size=$BATCH_SIZE")
if [[ -n "${TRAIN_STEPS:-}" ]]; then
    task_overrides+=("--steps=$TRAIN_STEPS" "--policy.scheduler_decay_steps=$TRAIN_STEPS")
fi
exec bash "$task_training_dir/train.sh" "$task_training_dir/configs/dualarm_fold_towel_no_rtc_full.json" "${task_overrides[@]}" "$@"
