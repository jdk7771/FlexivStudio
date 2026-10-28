#!/usr/bin/env bash
# Full-model training on two GPUs: global batch 32 = 16 per GPU, accumulation 1.
set -euo pipefail

task_training_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}"
export MASTER_PORT="${MASTER_PORT:-29519}"
task_num_gpus=2
task_global_batch_size=32
task_batch_size=$((task_global_batch_size / task_num_gpus))
exec bash "$task_training_dir/train.sh" "$task_training_dir/configs/dualarm_fold_towel_rtc_full.json" \
    "--parallelism.dp_replicate=$task_num_gpus" \
    --parallelism.dp_shard=1 \
    "--batch_size=$task_batch_size" \
    --accelerator.gradient_accumulation.steps=1 \
    "$@"
