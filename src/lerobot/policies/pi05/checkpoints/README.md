# PI05 checkpoints：区分视觉冻结与全量训练

## 新版 RTC：包含视觉参数训练

2026-10-27 从 `101.132.150.51:1024` 的 `/mnt/work/dakun/outputs/rtc_full/checkpoints/` 下载了 `015000`、`020000`、`025000`。每份保存的 `config.json` 和 `train_config.json` 均确认 `freeze_vision_encoder=false`、`train_expert_only=false`、`use_peft=false`、`rtc_training_max_delay=8`。

```text
rtc_full/
├── 015000/pretrained_model/
├── 020000/pretrained_model/
├── 025000/pretrained_model/
├── last -> 025000
├── tokenizer -> last/pretrained_model/tokenizer
├── SHA256SUMS
├── manifest.json
└── validation/                 # 离线观测、预测结果、运行日志与验证记录
```

三份共 27 个模型、配置、归一化和 tokenizer 文件，28,165,408,854 字节（约 26.23 GiB），逐文件 SHA256 与服务器一致。本次只下载部署所需的 `pretrained_model/`，不含优化器和其他 `training_state/`。

| step | model.safetensors SHA256 |
| ---- | ----------------------- |
| 15,000 | `5ead551ad1f238b61f1dbef047b3286afc4468652baff143b931b96acbe945cc` |
| 20,000 | `f980911da2ac9c05b78b1dcec544ff70ac1cb65882757375738a122a168e53c7` |
| 25,000 | `29aca1e61f5af0d6a712fdbbbf0a21170dab02252c32cf4d44714b7081fe3374` |

三份均在本机 RTX 5090 上严格加载并使用同一帧原始示教观测完成离线 RTC 推理，输出为数值有限的 `(50, 16)` 动作块；25k 另外通过非 RTC 同步推理检查。这验证模型加载和单次预测，不代表真机任务成功率。

RTC 部署默认选择本组 `last`，即本次下载的 25k；原始模型和训练配置保留服务器内容，本地 tokenizer 路径由部署入口覆盖。可以重新校验：

```bash
cd src/lerobot/policies/pi05/checkpoints/rtc_full
sha256sum -c SHA256SUMS
```

## 旧版 RTC：视觉冻结

本机保存服务器实验 `DualArm_FoldTowel_pi05_rtc_full_gpu5_6` （`freeze_vision_encoder=true`、`train_expert_only=false`、无 LoRA）的四个完整 checkpoint，以及 25,000 步的部署权重：

```text
checkpoints/
├── README.md
└── rtc_frozen_vision/            # 本地模型文件，Git 忽略
    ├── 025000/                  # 仅 pretrained_model/
    ├── 030000/
    ├── 040000/
    ├── 045000/
    ├── 050000/
    ├── last -> 050000
    ├── SHA256SUMS
    └── manifest.json
```

`030000`、`040000`、`045000`、`050000` 包含 `pretrained_model/`（权重、模型配置、实际训练配置、预处理和归一化统计）与 `training_state/`（优化器、调度器、随机状态、训练步数）。`050000` 同时是该实验最终 checkpoint，这四份总大小约 79 GiB。`025000` 仅下载完整 `pretrained_model/`，用于部署，不含恢复训练所需的 `training_state/`。

本地目录原名 `rtc_full`，现已改为 `rtc_frozen_vision`；各 checkpoint 原始权重、训练配置和校验值保持不变，`last` 仍指向旧实验的 `050000`。

来源：`101.132.150.51:1020`，服务器路径 `/mnt/work/dakun/outputs/DualArm_FoldTowel_pi05_rtc_full_gpu5_6/checkpoints/`。原始文件内容保持不变，服务器路径通过部署配置覆盖。

下载完成后，`rtc_frozen_vision/SHA256SUMS` 记录服务器原文件校验值，`manifest.json` 记录各保存点、文件大小和来源。可以在该目录重新核对：

```bash
cd src/lerobot/policies/pi05/checkpoints/rtc_frozen_vision
sha256sum -c SHA256SUMS
```

配套部署代码见 [deployment/README.md](../deployment/README.md)。
