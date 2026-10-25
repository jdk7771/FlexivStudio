# DualArm_FoldTowel：服务器上的数据、代码和训练指令

代码仓库：[jdk7771/FlexivStudio](https://github.com/jdk7771/FlexivStudio)。训练与本地部署共用 `main`；同步方式见 [项目入口](README.md)，历史训练状态见 [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md)。

数据：`/mnt/work/dakun/DualArm_FoldTowel_joint_rtc`。
代码：`/mnt/work/dakun/lerobot`。
任务文本：`DualArm_FoldTowel`。
现有 Python：`/mnt/work/dakun/conda-envs/lerobot/bin/python`。现有 PI05 base：`/mnt/work/dakun/models/pi05_base`；tokenizer：`/mnt/work/dakun/models/paligemma-3b-pt-224`。

## 当前训练配置

| 配置         | LoRA                               | 全量微调                                 |
| ------------ | ---------------------------------- | ---------------------------------------- |
| 脚本         | `train_dualarm_fold_towel_rtc.sh`  | `train_dualarm_fold_towel_rtc_full.sh`   |
| 默认物理 GPU | `4`                                | `6,7`（与对齐时服务器脚本一致）          |
| batch        | 16                                 | 每卡 8，全局 16                          |
| 学习率       | `1e-4`                             | `1e-5`                                   |
| 可训练范围   | 动作专家及投影 LoRA，rank/alpha 32 | 冻结视觉编码器，其余全量微调             |
| 默认输出后缀 | `DualArm_FoldTowel_pi05_rtc_lora`  | `DualArm_FoldTowel_pi05_rtc_full_gpu5_6` |

全量脚本保留旧输出目录和 job 名中的 `gpu5_6`，以兼容已有 checkpoint；实际 GPU 以 `CUDA_VISIBLE_DEVICES` 为准。两份脚本支持环境变量 `LEROBOT_PYTHON`、`DATASET_ROOT`、`MODEL_PATH`、`TOKENIZER_PATH`、`OUTPUT_DIR` 和 `CUDA_VISIBLE_DEVICES`。

## 首次安装环境（手动执行）

现有训练环境可以继续使用。新建环境时，安装脚本会在仓库创建独立 Python 3.12 `.venv`，使用 Torch 2.7.1 + CUDA 11.8、匹配的 TorchVision / TorchCodec；不改变服务器全局环境。

```bash
cd /mnt/work/dakun/lerobot
bash examples/flexiv/setup_server.sh
export LEROBOT_PYTHON="$PWD/.venv/bin/python"
```

安装版本参考：[PyTorch 官方旧版本安装说明](https://pytorch.org/get-started/previous-versions/)、[TorchCodec 兼容表](https://github.com/meta-pytorch/torchcodec#installing-torchcodec)。
训练脚本使用 TorchCodec 读取视频。安装脚本不会启动训练，也不会下载 pi05 模型权重。

## 查看训练命令 / 开始训练

先只打印命令，不启动训练：

```bash
bash examples/flexiv/train_dualarm_fold_towel_rtc.sh --print-command
bash examples/flexiv/train_dualarm_fold_towel_rtc_full.sh --print-command
```

你决定开始时，再手动执行：

```bash
bash examples/flexiv/train_dualarm_fold_towel_rtc.sh
```

LoRA 默认 π0.5 base + LoRA rank 32 / alpha 32，冻结视觉和 VLM，只适配动作专家及投影；BF16，梯度检查点。
单卡 batch 16，梯度累积 1，有效 batch 16；50,000 个训练循环，对应 50,000 次优化器更新。
每 2,500 步保存一个 checkpoint；学习率调度衰减周期同步设为 50,000 步，预热 1,000 步。
Joint Delta + Gripper Absolute，chunk 50；training RTC 最大已知前缀 8 帧，在 15 Hz 下约 0.53 秒。
按源 scene 留出最后 10% 场景验证，避免同一演示的切段同时出现在训练和验证中。
默认输出：`/mnt/work/dakun/outputs/DualArm_FoldTowel_pi05_rtc_lora`。

LoRA 默认从 Hub 读取 `lerobot/pi05_base`，缓存到 `/mnt/work/dakun/huggingface`；全量脚本默认使用本地 base。两者均默认使用本地 tokenizer；也可以覆盖模型和 tokenizer 目录：

```bash
MODEL_PATH=/mnt/work/dakun/models/pi05_base \
TOKENIZER_PATH=/mnt/work/dakun/models/paligemma-3b-pt-224 \
  bash examples/flexiv/train_dualarm_fold_towel_rtc.sh
```

开始训练前确认 GPU 使用情况和输出目录。默认 GPU 只反映脚本配置，不代表当前空闲。

## 恢复全量实验

使用原输出目录和完整 checkpoint，先打印命令检查：

```bash
CUDA_VISIBLE_DEVICES=6,7 \
  bash examples/flexiv/train_dualarm_fold_towel_rtc_full.sh --print-command \
    --resume=true \
    --config_path=/mnt/work/dakun/outputs/DualArm_FoldTowel_pi05_rtc_full_gpu5_6/checkpoints/015000/pretrained_model/train_config.json
```

将示例 step 替换为实际恢复点，并确认同级 `training_state/` 完整。检查后去掉 `--print-command` 才会启动。同步代码不会恢复优化器状态，也不会改变正在运行的训练进程。
