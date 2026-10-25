#!/usr/bin/env bash
# Manual launch only. --print-command prints the command without starting any GPU process.
set -euo pipefail

task_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
task_python="${LEROBOT_PYTHON:-/mnt/work/dakun/conda-envs/lerobot/bin/python}"
task_dataset_root="${DATASET_ROOT:-/mnt/work/dakun/DualArm_FoldTowel_joint_rtc}"
task_model_path="${MODEL_PATH:-lerobot/pi05_base}"
task_tokenizer_path="${TOKENIZER_PATH:-/mnt/work/dakun/models/paligemma-3b-pt-224}"
task_output="${OUTPUT_DIR:-/mnt/work/dakun/outputs/DualArm_FoldTowel_pi05_rtc_lora}"
task_print_only=false
if [[ "${1:-}" == --print-command ]]; then
    task_print_only=true
    shift
fi
if [[ ! -x "$task_python" ]]; then
    printf '%s\n' "Set LEROBOT_PYTHON to a valid Python executable first." >&2
    exit 1
fi

# Holding out complete source scenes prevents leakage between fragments of one demonstration.
task_dataset_config="$("$task_python" - "$task_dataset_root" <<'PY'
import json
import math
import sys
from pathlib import Path
root = Path(sys.argv[1])
manifest = json.loads((root / 'meta/conversion_manifest.json').read_text())
if manifest['status'] != 'complete':
    raise SystemExit('Dataset conversion/statistics are incomplete.')
scenes = [s for s in manifest['scenes'] if s['episodes']]
total = manifest['total_episodes']
holdout_scenes = min(max(1, math.ceil(len(scenes) * 0.1)), len(scenes) - 1)
holdout = sum(s['episodes'] for s in scenes[-holdout_scenes:]) if holdout_scenes else 0
split = (holdout - 0.5) / total if holdout else 0.0
print(manifest['repo_id'])
print(split)
PY
)"
mapfile -t task_dataset_fields <<< "$task_dataset_config"
task_repo_id="${task_dataset_fields[0]}"
task_eval_split="${task_dataset_fields[1]}"
task_eval_steps=1000
if [[ "$task_eval_split" == 0.0 ]]; then
    task_eval_steps=0
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
export OMP_NUM_THREADS=4
export HF_HOME="${HF_HOME:-/mnt/work/dakun/huggingface}"
export TOKENIZERS_PARALLELISM=false
# pip's NVIDIA wheels are installed inside this environment rather than system-wide.
task_site_packages="$($task_python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
task_nvidia_libs="$(find "$task_site_packages/nvidia" -mindepth 2 -maxdepth 2 -type d -name lib -printf '%p:' 2>/dev/null || true)"
export LD_LIBRARY_PATH="${task_nvidia_libs}${LD_LIBRARY_PATH:-}"
cd "$task_repo_root"
task_command=(
    "$task_python" -m lerobot.scripts.lerobot_train
    "--dataset.repo_id=$task_repo_id"
    "--dataset.root=$task_dataset_root"
    "--dataset.eval_split=$task_eval_split"
    --dataset.video_backend=torchcodec
    --policy.type=pi05
    "--policy.pretrained_path=$task_model_path"
    "--policy.text_tokenizer_name=$task_tokenizer_path"
    --policy.device=cuda
    --policy.dtype=bfloat16
    --policy.push_to_hub=false
    --policy.use_relative_actions=true
    '--policy.relative_exclude_joints=["gripper"]'
    --policy.chunk_size=50
    --policy.rtc_training_max_delay=8
    --policy.freeze_vision_encoder=true
    --policy.train_expert_only=true
    --policy.gradient_checkpointing=true
    --policy.optimizer_lr=0.0001
    --policy.optimizer_weight_decay=0.00001
    --policy.scheduler_decay_steps=50000
    --peft.method_type=LORA
    --peft.r=32
    --peft.lora_alpha=32
    --parallelism.dp_replicate=1
    --accelerator.mixed_precision=bf16
    --accelerator.gradient_accumulation.steps=1
    --batch_size=16
    --num_workers=4
    --steps=50000
    --save_freq=2500
    --log_freq=50
    --env_eval_freq=0
    "--eval_steps=$task_eval_steps"
    --max_eval_samples=256
    --wandb.enable=false
    "--output_dir=$task_output"
    --job_name=DualArm_FoldTowel_pi05_rtc_lora
)
if "$task_print_only"; then
    printf 'CUDA_VISIBLE_DEVICES=%q ' "$CUDA_VISIBLE_DEVICES"
    printf '%q ' "${task_command[@]}" "$@"
    printf '\n'
    exit 0
fi
exec "${task_command[@]}" "$@"
