# PI05 checkpoint 保存计划与训练来源

本目录保存 `rtc_full` 和本次 `rtc_full_finetune` 的新训练产物，并管理模型来源与部署分组说明。
两份配置分别使用独立的实验目录；每个实验目录内由训练器创建 `checkpoints/<step>/`。
历史服务器实验仍位于 `/mnt/data/dakun/traingProcess/projects/lerobot/`。
机器人机器下载部署权重时，可保留训练目录结构，也可按 `<实验名>/<step>/pretrained_model/` 存放。
权重目录由 Git 忽略；同步仓库只更新源码和说明，不重命名已有权重、修改保存配置或删除 checkpoint。

## 本次 RTC 全量微调：DualArm_FoldTowel_pi05_rtc_full_finetune

训练入口为 [`train_dualarm_fold_towel_rtc_full_finetune.sh`](../training/train_dualarm_fold_towel_rtc_full_finetune.sh)，
配置为 [`dualarm_fold_towel_rtc_full_finetune.json`](../training/configs/dualarm_fold_towel_rtc_full_finetune.json)。
GPU `6,7`，从 `pi05_base` 开始全模型训练，无 LoRA、无测评，双卡全局 batch 32。
本次使用 FP32 模型、BF16 混合精度，学习率 `2.5e-5`、权重衰减 `0.01`、关闭梯度检查点。
与历史 `rtc_full` 的参数对照见 [训练说明](../training/README.md)。

以下是计划保存结构，尚不表示这些 checkpoint 已生成：

```text
/mnt/data/dakun/workspace/lerobot/src/lerobot/policies/pi05/checkpoints/
└── DualArm_FoldTowel_pi05_rtc_full_finetune/
    └── checkpoints/
        ├── 010000/
        ├── 020000/
        ├── 030000/
        ├── 040000/
        ├── 050000/
        └── last -> 050000        # 训练完成后；训练中指向最新完整保存点
```

每个步数目录包含 `pretrained_model/`（权重、模型与训练配置、预处理和归一化状态）
以及 `training_state/`（优化器、调度器、随机状态和训练步数）。
`steps=50000`、`save_freq=10000`，五份完整 checkpoint 全部保留，训练器没有自动轮换删除。
本次不采用历史实验“只保留最后两份”的清理规则。

部署入口支持本次原生训练结构
`pi05/checkpoints/DualArm_FoldTowel_pi05_rtc_full_finetune/checkpoints/<step>/pretrained_model/`，
也支持机器人机器上原有的 `<实验名>/<step>/pretrained_model/` 下载结构。
目录菜单与 `run.sh <实验名> <step>`、`run_no_rtc.sh <实验名> <step>` 均可选择这两种布局。
tokenizer 使用训练时同一份 PaliGemma tokenizer，可放入该 `pretrained_model/tokenizer/`，
或通过部署命令的 `--tokenizer` 指定机器人机器上的路径。
完整续训保留服务器的 `training_state/`，使用该 checkpoint 的 `train_config.json`；
只复制部署权重不能恢复优化器与调度器状态。
选择新模型的命令见 [部署说明](../deployment/README.md)。

## 历史三组模型

以下记录对应原机器人机器上的下载结果，不代表当前服务器源码目录含有这些权重。
原部署记录中的步数和校验结果保留如下，实际可用步数以本地文件及 `last` 链接为准。

| 本地组名 | 配置含义 | 历史来源（相对服务器 `/mnt/work/dakun/logs/projects/lerobot`） | 部署入口 |
| --- | --- | --- | --- |
| `DualArm_FoldTowel_pi05_rtc_frozen_vision` | RTC，冻结视觉，其余参数训练，无 LoRA | `DualArm_FoldTowel_pi05_rtc_full_gpu5_6` | `run.sh` 显式选择该组 |
| `DualArm_FoldTowel_pi05_rtc_full` | RTC，包含视觉的全模型训练，无 LoRA | `rtc_full` | `run.sh` 默认组 |
| `DualArm_FoldTowel_pi05_no_rtc_full` | 无 RTC，包含视觉的全模型训练，无 LoRA | `DualArm_FoldTowel_pi05_no_rtc_full_gpu7_6_5_4` | `run_no_rtc.sh` 默认组 |

另有 [openpi joint 原模型训练](../../openpi/training/README.md)，实验名为
`openpi_pi05_dualarm_fold_towel_rtc_full`，使用 openpi 自己的训练和部署入口。
新增 [openpi π0.5 pose delta 无 RTC 训练](../../openpi/training_pose_delta/README.md) 使用独立 pose 输出目录。
两个 openpi 入口均不作为本目录三个 LeRobot checkpoint 组之一，也不能与 LeRobot 权重混放。

## RTC 全模型：DualArm_FoldTowel_pi05_rtc_full

原下载记录：2026-10-27 从 `101.132.150.51:1024` 的
`/mnt/work/dakun/logs/projects/lerobot/rtc_full/checkpoints/` 下载 `015000`、`020000`、`025000`。
三份保存配置均为 `freeze_vision_encoder=false`、`train_expert_only=false`、
`use_peft=false`、`rtc_training_max_delay=8`；历史训练为四卡每卡 8、全局 batch 32。
`training/configs/dualarm_fold_towel_rtc_full.json` 已按保存配置恢复历史训练参数：
四卡每卡 8、BF16、学习率 `1e-5`、权重衰减 `1e-5`、开启梯度检查点，
每 2,500 步保存、每 1,000 步验证，`eval_split=0.08536585365853659`。
核对来源为服务器 `rtc_full/checkpoints/025000/pretrained_model/train_config.json`，
并与 `022500` 保存配置交叉核对。完整配置 SHA256 记录在 [训练说明](../training/README.md)。
Git 提交 `fec5968` 对当前模板的后续修改不会改变历史权重当时使用的配置。
本次 GPU 6、7 的新实验使用独立的 `rtc_full_finetune` 配置与目录。

```text
DualArm_FoldTowel_pi05_rtc_full/
├── 015000/pretrained_model/
├── 020000/pretrained_model/
├── 025000/pretrained_model/
├── last -> 025000                # 原下载记录中的链接目标
├── tokenizer -> last/pretrained_model/tokenizer
├── SHA256SUMS
├── manifest.json
└── validation/
```

原校验记录：27 个模型、配置、归一化和 tokenizer 文件，共 28,165,408,854 字节，
逐文件 SHA256 与来源一致；这里只下载 `pretrained_model/`，不含 `training_state/`。

| step | model.safetensors SHA256 |
| --- | --- |
| 15,000 | `5ead551ad1f238b61f1dbef047b3286afc4468652baff143b931b96acbe945cc` |
| 20,000 | `f980911da2ac9c05b78b1dcec544ff70ac1cb65882757375738a122a168e53c7` |
| 25,000 | `29aca1e61f5af0d6a712fdbbbf0a21170dab02252c32cf4d44714b7081fe3374` |

原本地验证记录：三份均在 RTX 5090 上严格加载，使用同一观测完成离线 RTC 推理，
输出为有限的 `(50, 16)` 动作块；25k 另通过非 RTC 同步推理检查。
这是既有单次推理验证记录，不代表本次整理重新执行推理或验证真机成功率。

```bash
cd src/lerobot/policies/pi05/checkpoints/DualArm_FoldTowel_pi05_rtc_full
sha256sum -c SHA256SUMS
```

## RTC 冻结视觉：DualArm_FoldTowel_pi05_rtc_frozen_vision

来源为 `101.132.150.51:1020` 的
`/mnt/work/dakun/logs/projects/lerobot/DualArm_FoldTowel_pi05_rtc_full_gpu5_6/checkpoints/`。
历史服务器目录名含 `full`，但保存配置实际为 `freeze_vision_encoder=true`、
`train_expert_only=false`、无 LoRA，故本地统一使用 `rtc_frozen_vision` 后缀。
早期本地曾使用短名 `rtc_full`、`rtc_frozen_vision`；当前部署源码采用上面的完整组名。

原下载记录包含 `025000`、`030000`、`040000`、`045000`、`050000`，以及
`last -> 050000`、`SHA256SUMS`、`manifest.json`。
25k 只有部署用的 `pretrained_model/`；30k/40k/45k/50k 同时包含模型和
`training_state/`（优化器、调度器、随机状态及步数），这四份共约 79 GiB。
模型内容、保存配置及校验值继续使用原值；原路径由部署入口按机器环境覆盖。

```bash
cd src/lerobot/policies/pi05/checkpoints/DualArm_FoldTowel_pi05_rtc_frozen_vision
sha256sum -c SHA256SUMS
```

## 无 RTC 全模型：DualArm_FoldTowel_pi05_no_rtc_full

历史服务器来源为
`/mnt/work/dakun/logs/projects/lerobot/DualArm_FoldTowel_pi05_no_rtc_full_gpu7_6_5_4/checkpoints/`。
保存配置确认 `freeze_vision_encoder=false`、`train_expert_only=false`、
`use_peft=false`、`rtc_training_max_delay=0`；32.5k 保存配置另确认
`dataset.eval_split=0`、`eval_steps=0`。当前新训练模板启用留出集和每 1000 步验证，
因此不能用当前模板声称完全复现这次历史训练。

`deployment/deploy_config_no_rtc.json` 默认选择本地该组的 `025000/pretrained_model/`。
服务器上的步数不能直接推断本地已下载步数；部署菜单按本地实际模型文件发现。
非 RTC 入口也可以显式加载 RTC 训练组关闭 RTC 推理，但这不会改变该模型的训练来源。

## 配置与恢复原则

- 新训练统一命名；历史服务器输出目录和 checkpoint 保存配置保留原值。
- 恢复训练使用原 checkpoint 的 `train_config.json` 与完整训练状态；部署权重不等于可恢复训练的 checkpoint。
- `manifest.json`、`SHA256SUMS` 是历史来源和内容证据，不为统一名称而改写。
- [训练 README](../training/README.md) 列出本次入口、原三组模板与历史设置差异；[部署 README](../deployment/README.md) 提供模型选择命令。
