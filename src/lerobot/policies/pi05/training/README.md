# PI05 训练入口

本次新实验使用 `rtc_full_finetune`。`rtc_full` 保留历史 RTC 全量实验的训练参数，
对应机器人机器上的 `checkpoints/DualArm_FoldTowel_pi05_rtc_full`；两份配置分别管理历史实验与本次实验。

## rtc_full 与 rtc_full_finetune

| 参数 | `rtc_full`：历史实验参数 | `rtc_full_finetune`：本次新实验 |
| --- | --- | --- |
| 配置 | [dualarm_fold_towel_rtc_full.json](configs/dualarm_fold_towel_rtc_full.json) | [dualarm_fold_towel_rtc_full_finetune.json](configs/dualarm_fold_towel_rtc_full_finetune.json) |
| Bash | [train_dualarm_fold_towel_rtc_full.sh](train_dualarm_fold_towel_rtc_full.sh) | [train_dualarm_fold_towel_rtc_full_finetune.sh](train_dualarm_fold_towel_rtc_full_finetune.sh) |
| 默认物理 GPU | `0,1,2,3` | `6,7` |
| 每卡 batch × 卡数 | 8 × 4 | 16 × 2 |
| 全局 batch / 梯度累积 | 32 / 1 | 32 / 1 |
| 模型 dtype / 混合精度 | BF16 / BF16 | FP32 / BF16 |
| 梯度检查点 | 开启 | 关闭 |
| 学习率 | `1e-5` | `2.5e-5` |
| 权重衰减 | `1e-5` | `0.01` |
| warmup / decay steps | 1,000 / 50,000 | 1,000 / 50,000 |
| 训练步数 | 50,000 | 50,000 |
| checkpoint 保存间隔 | 2,500 | 10,000 |
| 验证集 / 验证间隔 | `eval_split=0.08536585365853659` / 1,000 steps，最多 256 样本 | 不划分验证集，不测评 |
| 默认端口 | `29519` | `29527` |
| 实验目录名 | `DualArm_FoldTowel_pi05_rtc_full` | `DualArm_FoldTowel_pi05_rtc_full_finetune` |

两者均从 `pi05_base` 开始训练视觉、VLM 和动作专家，无 LoRA，RTC delay 8、chunk 50；
动作使用相对关节角与绝对夹爪宽度。双卡或四卡并行度、每卡 batch 都写在各自 JSON 中，
Bash 选择 GPU 和端口。`CUDA_VISIBLE_DEVICES`、`MASTER_PORT` 和显式 CLI 参数可覆盖默认值。

历史参数核对来源：服务器
`/mnt/data/dakun/traingProcess/projects/lerobot/rtc_full/checkpoints/025000/pretrained_model/train_config.json`，
并与 `022500` 保存配置交叉核对。
25k 配置文件 SHA256 为 `6dead182d6920553450956fe9e612b618c01b044e1f870c8eed34c1ceea007b9`。
`rtc_full` 模板复原这些训练参数，机器路径和新训练输出名称适配当前工作区。
历史运行的原名为 `rtc_full`，历史目录与 checkpoint 保存配置继续保留。

Git 提交 `fec5968` 曾把 `rtc_full` 模板改成新训练默认值，随后复制的 `rtc_full_finetune`
一度与它重复。现在 `rtc_full` 按保存配置恢复历史参数；本次新设置只由 `rtc_full_finetune` 管理。
本次 `scheduler_decay_steps=50000` 保留当前设置。恢复旧训练始终使用 checkpoint 保存的实际配置。

## 本次 GPU 6、7 新训练

```bash
cd /mnt/data/dakun/workspace/lerobot

# 预览，不加载模型或启动训练
bash src/lerobot/policies/pi05/training/train_dualarm_fold_towel_rtc_full_finetune.sh --print-command

# 开始新训练
bash src/lerobot/policies/pi05/training/train_dualarm_fold_towel_rtc_full_finetune.sh
```

`resume=false`，从基础模型重新微调；`dataset.eval_split=0`、`eval_steps=0`、`env_eval_freq=0`，
全部数据用于训练。`save_checkpoint=true`、`save_freq=10000`，训练 50,000 steps。

按当前指定位置，`output_dir` 为：

```text
/mnt/data/dakun/workspace/lerobot/src/lerobot/policies/pi05/checkpoints/DualArm_FoldTowel_pi05_rtc_full_finetune/
```

训练器在 `output_dir/checkpoints/` 下保存 `010000`、`020000`、`030000`、`040000`、`050000`；
每份含 `pretrained_model/` 和完整 `training_state/`，`last` 指向最新完整保存点。
本次五份全部保留，不应用历史实验只保留最后两份的清理规则。
输出目录已存在时，新训练会拒绝覆盖；重复实验可用 `OUTPUT_DIR` 指定另一个目录。

中断后按实际已有步数续训，例如 20k：

```bash
CUDA_VISIBLE_DEVICES=6,7 MASTER_PORT=29527 \
bash src/lerobot/policies/pi05/training/train.sh \
  /mnt/data/dakun/workspace/lerobot/src/lerobot/policies/pi05/checkpoints/DualArm_FoldTowel_pi05_rtc_full_finetune/checkpoints/020000/pretrained_model/train_config.json \
  --resume=true
```

## 其他保留入口

| 简称 | Bash / JSON 后缀 | 含义 | 默认 GPU | 每卡 batch × 卡数 |
| --- | --- | --- | --- | --- |
| `rtc_full` | `dualarm_fold_towel_rtc_full` | 历史 RTC 全模型训练参数 | `0,1,2,3` | 8 × 4 |
| `rtc_frozen_vision` | `dualarm_fold_towel_rtc_frozen_vision` | RTC，冻结视觉，其余参数训练 | `6,7` | 8 × 2 |
| `no_rtc_full` | `dualarm_fold_towel_no_rtc_full` | 无 RTC，全模型训练 | `3,2,1,0` | 8 × 4 |

配置在 `configs/<后缀>.json`，脚本为 `train_<后缀>.sh`。
旧 LoRA、LoRA batch 8 与 `rtc_unfrozen_vision` 入口已移出当前模板，原配置可在 Git 历史中查阅。
`rtc_unfrozen_vision` 原总 batch 16，与本次总 batch 32 的实验分开记录。

`rtc_full` 新运行也保存到本仓库 `pi05/checkpoints/DualArm_FoldTowel_pi05_rtc_full/`，
历史模型保留在各机器原目录。冻结视觉和无 RTC 模板仍输出到
`/mnt/data/dakun/traingProcess/projects/lerobot/<实验名>/`。
冻结视觉与无 RTC 模板均启用留出集和每 1,000 步验证；无 RTC 历史 32.5k 保存配置实际关闭了验证，
因此其当前模板只代表新运行设置。

## 历史权重与续训

| 机器人机器上的组名 | 服务器历史实验目录（相对 `traingProcess/projects/lerobot`） | 已核实的历史设置 |
| --- | --- | --- |
| `DualArm_FoldTowel_pi05_rtc_full` | `rtc_full` | 25k：全模型 RTC，4 卡 × batch 8 |
| `DualArm_FoldTowel_pi05_rtc_frozen_vision` | `DualArm_FoldTowel_pi05_rtc_full_gpu5_6` | 名含 full，实际冻结视觉，2 卡 × batch 8 |
| `DualArm_FoldTowel_pi05_no_rtc_full` | `DualArm_FoldTowel_pi05_no_rtc_full_gpu7_6_5_4` | 32.5k：全模型无 RTC，关闭验证 |

续训使用原 checkpoint 的 `pretrained_model/train_config.json` 与同级 `training_state/`。
只下载部署权重不能恢复优化器、调度器与随机状态。
如果保存配置里的旧机器路径已失效，通过 `DATASET_ROOT`、`MODEL_PATH`、`TOKENIZER_PATH`
和 `--output_dir=...` 显式覆盖到当前实验目录。改变 GPU 数量时一并调整并行度与每卡 batch；
即使总 batch 相同，也不保证采样顺序逐位一致。
来源与保存结构见 [checkpoint 说明](../checkpoints/README.md)，模型选择见 [部署说明](../deployment/README.md)。

## 环境、监控与目录

| 文件 | 用途 |
| --- | --- |
| `train.sh` | 公共 JSON 启动器、机器路径覆盖、`--print-command` |
| `configs/`、`train_dualarm_*.sh` | 四套具名实验与对应快捷入口 |
| `setup_server.sh`、`server-constraints.txt` | 新环境安装与依赖约束 |
| `monitor_loss.py`、`loss_dashboard.html` | loss 采样、曲线和页面 |

现有 PI05 Python 为 `/mnt/data/dakun/envs/lerobot/bin/python`；无需重建环境。
数据在 `/mnt/data/dakun/datasets/DualArm_FoldTowel_joint_rtc`，base 和 tokenizer 分别在
`/mnt/data/dakun/model/pi05_base`、`/mnt/data/dakun/model/paligemma-3b-pt-224`。
`LEROBOT_PYTHON` 可选择解释器；环境索引见 `/mnt/data/dakun/envs/repo-environments.tsv`。

监控默认对应本次 `rtc_full_finetune`；显式指定其他实验用 `--output-dir`：

```bash
/mnt/data/dakun/envs/lerobot/bin/python src/lerobot/policies/pi05/training/monitor_loss.py --serve
```

默认端口 8766，可通过 VS Code 转发。采样曲线与终端训练窗口均值口径不同，停止监控不会停止训练。
旧目录迁移备份在 `/mnt/data/dakun/traingProcess/projects/lerobot/maintenance/`。

官方 OpenPI 的 [joint 训练](../../openpi/training/README.md) 与
[pose 训练](../../openpi/training_pose_delta/README.md) 使用独立模型实现、配置与
`/mnt/data/dakun/envs/openpi` 环境。其 checkpoint 使用 OpenPI 自己的部署入口。
历史实验记录见 [EXPERIMENT_LOG.md](../../../../../examples/flexiv/docs/EXPERIMENT_LOG.md)。
