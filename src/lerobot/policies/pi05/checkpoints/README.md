# RTC 全量微调 checkpoints

本机保存服务器实验 `DualArm_FoldTowel_pi05_rtc_full_gpu5_6` 的四个完整 checkpoint，以及 25,000 步的部署权重：

```text
checkpoints/
├── README.md
└── rtc_full/                     # 本地模型文件，Git 忽略
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

来源：`101.132.150.51:1020`，服务器路径 `/mnt/work/dakun/outputs/DualArm_FoldTowel_pi05_rtc_full_gpu5_6/checkpoints/`。原始文件内容保持不变，服务器路径通过部署配置覆盖。

下载完成后，`rtc_full/SHA256SUMS` 记录服务器原文件校验值，`manifest.json` 记录各保存点、文件大小和来源。可以在该目录重新核对：

```bash
cd src/lerobot/policies/pi05/checkpoints/rtc_full
sha256sum -c SHA256SUMS
```

配套部署代码见 [deployment/README.md](../deployment/README.md)。
