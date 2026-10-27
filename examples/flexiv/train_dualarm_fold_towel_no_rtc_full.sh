#!/usr/bin/env bash
# Compatibility entry point; experiment parameters live in policies/pi05/training/configs.
set -euo pipefail

task_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "$task_repo_root/src/lerobot/policies/pi05/training/train_dualarm_fold_towel_no_rtc_full.sh" "$@"
