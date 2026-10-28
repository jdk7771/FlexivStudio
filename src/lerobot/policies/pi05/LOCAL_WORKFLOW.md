# PI05 训练、权重与部署索引

本次训练使用 `rtc_full_finetune`，GPU `6,7`，双卡总 batch 32、50,000 steps、无测评，
每 10,000 steps 保存并保留一份完整 checkpoint。`rtc_full` 保留历史 RTC 全量实验参数。

| 实验 | 对应 Bash / JSON 后缀 | checkpoint 组名 | 训练设置 |
| --- | --- | --- | --- |
| `rtc_full_finetune`（本次） | `dualarm_fold_towel_rtc_full_finetune` | `DualArm_FoldTowel_pi05_rtc_full_finetune` | 2×16，FP32 模型 / BF16 混合精度，lr `2.5e-5`，无验证，10k 保存 |
| `rtc_full`（历史参数） | `dualarm_fold_towel_rtc_full` | `DualArm_FoldTowel_pi05_rtc_full` | 4×8，BF16，lr `1e-5`，开启验证，2.5k 保存 |
| `rtc_frozen_vision` | `dualarm_fold_towel_rtc_frozen_vision` | `DualArm_FoldTowel_pi05_rtc_frozen_vision` | 冻结视觉，其余训练，RTC delay 8 |
| `no_rtc_full` | `dualarm_fold_towel_no_rtc_full` | `DualArm_FoldTowel_pi05_no_rtc_full` | 全模型，RTC delay 0 |

Bash 为 `training/train_<后缀>.sh`，JSON 为 `training/configs/<后缀>.json`。
两份全量 RTC 配置的参数差异与历史核对来源见 [training/README.md](training/README.md)。

| 目录 | 职责 | 说明 |
| --- | --- | --- |
| `training/` | 实验配置、训练入口、环境和 loss 监控 | [训练说明](training/README.md) |
| `checkpoints/` | 本次新训练产物、来源索引和本地部署权重 | [权重说明](checkpoints/README.md) |
| `deployment/` | RTC / 非 RTC / 高频状态记录、回位和回放 | [部署说明](deployment/README.md) |

新 `rtc_full` 和 `rtc_full_finetune` 按当前指定位置保存在
`/mnt/data/dakun/workspace/lerobot/src/lerobot/policies/pi05/checkpoints/<实验名>/checkpoints/<step>/`。
部署入口支持这个训练布局和机器人机器上原有的 `<实验名>/<step>/pretrained_model/` 下载布局。
权重目录由 Git 忽略。

机器人机器上的 `DualArm_FoldTowel_pi05_rtc_full` 来源为服务器历史 `rtc_full` 实验。
历史 `DualArm_FoldTowel_pi05_rtc_full_gpu5_6` 实际冻结视觉，归入 `rtc_frozen_vision` 组。
历史服务器实验仍位于 `/mnt/data/dakun/traingProcess/projects/lerobot/`，保存配置和权重保留原值。

RTC 默认部署组仍为历史 `DualArm_FoldTowel_pi05_rtc_full/last`；
新训练完成后，显式选择 `DualArm_FoldTowel_pi05_rtc_full_finetune` 和所需步数。
非 RTC 默认组为 `DualArm_FoldTowel_pi05_no_rtc_full/025000`。
实际可用步数以所在机器文件为准。

官方 OpenPI 的 [joint 训练](../openpi/training/README.md) 和
[pose 训练](../openpi/training_pose_delta/README.md) 使用独立实现、环境与部署入口，
不与以上 LeRobot PI05 checkpoint 混放。
