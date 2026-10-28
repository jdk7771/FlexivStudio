# π0.5 训练：配置定义实验，bash 负责启动

每次训练应能对应一份完整的最终配置。这里使用 LeRobot 原生 JSON 配置保存数据、模型、LoRA、学习率、batch、step 和验证参数；bash 只选择 Python 环境、GPU 和启动进程数。

```text
training/
├── configs/                           # 新训练的参数模板
│   ├── dualarm_fold_towel_rtc_lora.json
│   ├── dualarm_fold_towel_rtc_frozen_vision.json
│   ├── dualarm_fold_towel_rtc_unfrozen_vision.json
│   ├── dualarm_fold_towel_rtc_full.json
│   ├── dualarm_fold_towel_rtc_lora_bs8.json
│   └── dualarm_fold_towel_no_rtc_full.json
├── train.sh                           # 公共入口：JSON → 单进程或 torchrun
├── train_dualarm_fold_towel_rtc.sh     # LoRA / GPU 4 快捷入口
├── train_dualarm_fold_towel_rtc_frozen_vision.sh # RTC，冻结视觉编码器 / GPU 6,7
├── train_dualarm_fold_towel_rtc_unfrozen_vision.sh # RTC，全部参数训练 / GPU 6,7
├── train_dualarm_fold_towel_rtc_full.sh # rtc_full，全部参数训练 / GPU 4,5，每卡 batch 16
├── train_dualarm_fold_towel_rtc_lora_bs8.sh # LoRA batch 8 / GPU 7
├── train_dualarm_fold_towel_no_rtc_full.sh # 无 RTC 全量 / GPU 7,6,5,4
├── setup_server.sh                    # 服务器新建环境
└── server-constraints.txt
```

## 现有配置

E1/E2/E3 配置按 [实验记录](../../../../../examples/flexiv/docs/EXPERIMENT_LOG.md) 中保存的 `train_config.json` 提取，另提供 RTC 不冻结视觉编码器和四卡无 RTC 全量训练配置。它们用于**新训练**，不包含旧训练的恢复状态。

| 模板                                   | 对应设置                               | 每卡 batch / 全局 batch | 学习率 | 默认 GPU（快捷脚本） |
| -------------------------------------- | -------------------------------------- | ----------------------- | ------ | -------------------- |
| `dualarm_fold_towel_rtc_lora.json`     | E1，动作专家和投影 LoRA，rank/alpha 32 | 16 / 16                 | `1e-4` | `4`                  |
| `dualarm_fold_towel_rtc_frozen_vision.json` | E2 的冻结方式：冻结视觉编码器，其余参数训练，无 LoRA | 8 / 16 | `1e-5` | `6,7` |
| `dualarm_fold_towel_rtc_unfrozen_vision.json` | RTC，视觉编码器、VLM 和动作专家全部训练，无 LoRA | 8 / 16 | `1e-5` | `6,7` |
| `dualarm_fold_towel_rtc_full.json` | 本次 `rtc_full`，视觉编码器、VLM 和动作专家全部训练，无 LoRA | 16 / 32 | `1e-5` | `4,5` |
| `dualarm_fold_towel_rtc_lora_bs8.json` | E3，同 E1，batch 8                     | 8 / 8                   | `1e-4` | `7`                  |
| `dualarm_fold_towel_no_rtc_full.json` | 无 RTC，全模型训练，包含视觉编码器     | 8 / 32                  | `1e-5` | `7,6,5,4`            |

共同参数：BF16、梯度检查点、50,000 step、每 2,500 step 保存、每 1,000 step 验证最多 256 个样本。动作是关节相对当前观测、夹爪绝对值，chunk 50。所有 RTC 配置的 `policy.rtc_training_max_delay=8`；四卡无 RTC 配置设为 `policy.rtc_config=null`、`policy.rtc_training_max_delay=0`，关闭 LoRA 并训练全部模型参数。

RTC 的两组非 LoRA 配置通过 `policy.freeze_vision_encoder` 区分：`frozen_vision=true`，`unfrozen_vision=false`；两组的 `train_expert_only=false`、`use_peft=false`、`peft=null`。除冻结开关、实验名和输出目录外，两份 JSON 完全相同，以便比较。它们均从 `pi05_base` 开始，分别保存到 `/mnt/work/dakun/outputs/DualArm_FoldTowel_pi05_rtc_frozen_vision` 和 `/mnt/work/dakun/outputs/DualArm_FoldTowel_pi05_rtc_unfrozen_vision`，可通过 `OUTPUT_DIR` 覆盖。

原来冻结视觉编码器的 RTC 实验使用明确命名的 `rtc_frozen_vision` 入口；服务器历史输出目录保留原名，续训仍读取其 `train_config.json`；本地旧模型目录由 `checkpoints/rtc_full` 改为 `checkpoints/rtc_frozen_vision`，旧版部署时显式选择 `rtc_frozen_vision`。新四卡全模型实验使用 `checkpoints/rtc_full`，本次从 1024 服务器下载了 15k/20k/25k，RTC 部署默认选择新组 25k；两组模型不按相同步数混放。`rtc_frozen_vision` 和 `rtc_unfrozen_vision` 两卡入口默认使用 GPU `6,7`，可用 `CUDA_VISIBLE_DEVICES` 覆盖；冻结版默认端口 `29517`，不冻结版默认 `29519`，均可通过 `MASTER_PORT` 覆盖。

按本次实验命名，`train_dualarm_fold_towel_rtc_full.sh` 和 `dualarm_fold_towel_rtc_full.json` 现在用于 **GPU 4,5、不冻结视觉编码器** 的 RTC 全模型训练：`job_name=rtc_full`，输出 `/mnt/work/dakun/outputs/rtc_full`，每卡 batch 16、全局 batch 32，默认端口 `29519`。从 `pi05_base` 开始训练 50,000 step，RTC delay 8，数据划分和学习率沿用上述 RTC 对照配置。

已有 `rtc_full` checkpoint 原先采用四卡、每卡 batch 8。切换为双卡、每卡 batch 16 续训时，从 checkpoint 配置启动并显式覆盖并行度和 batch；全局 batch 仍为 32，优化器、调度器和 step 从 checkpoint 恢复。卡数变化后，不保证随机采样与原四卡运行逐位一致：

```bash
CUDA_VISIBLE_DEVICES=4,5 MASTER_PORT=29519 \
bash src/lerobot/policies/pi05/training/train.sh \
  /mnt/work/dakun/outputs/rtc_full/checkpoints/last/pretrained_model/train_config.json \
  --resume=true --parallelism.dp_replicate=2 --batch_size=16
```

四卡无 RTC 配置固定使用物理 GPU `7,6,5,4`，每卡 batch 8、梯度累积 1，全局 batch 32；默认端口为 `29518`。输出目录为 `/mnt/work/dakun/outputs/DualArm_FoldTowel_pi05_no_rtc_full_gpu7_6_5_4`，可用 `OUTPUT_DIR` 指定另一次实验的目录。快捷脚本兼容 `BATCH_SIZE` 和 `TRAIN_STEPS` 环境变量；显式命令行参数优先。

数据为 `local/DualArm_FoldTowel_joint_rtc`，123 个 episode。`dataset.eval_split=0.08536585365853659` 与原转换清单按最后 10% 源 scene 留出验证集的结果一致：112 个训练 episode、11 个验证 episode。这是该数据集的固定划分；更换数据后应一起检查 `repo_id`、`root`、`eval_split` 和 `eval_steps`。

## 先看命令，再启动

从仓库根目录执行。`--print-command` 只读 JSON 并打印命令，不导入训练器、不读取训练数据或加载权重：

```bash
pi05_training=src/lerobot/policies/pi05/training
bash "$pi05_training/train_dualarm_fold_towel_rtc.sh" --print-command
bash "$pi05_training/train_dualarm_fold_towel_rtc_frozen_vision.sh" --print-command
bash "$pi05_training/train_dualarm_fold_towel_rtc_unfrozen_vision.sh" --print-command
bash "$pi05_training/train_dualarm_fold_towel_rtc_full.sh" --print-command
bash "$pi05_training/train_dualarm_fold_towel_rtc_lora_bs8.sh" --print-command
bash "$pi05_training/train_dualarm_fold_towel_no_rtc_full.sh" --print-command
```

四卡无 RTC 训练启动后，可在另一个终端监控训练 loss：

```bash
python src/lerobot/policies/pi05/training/monitor_loss.py
```

脚本自动找到对应的 rank 0 进程，每 30 秒读取进度和一个 batch 的训练 loss，持续保存到当前输出目录的 `train_loss_monitor.csv`。重启监控会继续保留已有记录。采样值是 rank 0 的单 batch loss，与终端输出的多卡训练窗口均值分别显示。验证 loss 另外保存在 `eval_loss_monitor.csv`，关闭验证仍会继续采集训练 loss。`Ctrl+C` 只停止监控，不影响训练。使用 `--once` 可只查看一次；`--interval=60` 可调整采样间隔。

需要浏览器实时曲线和完整训练配置时运行：

```bash
python src/lerobot/policies/pi05/training/monitor_loss.py --serve
```

然后打开 `http://127.0.0.1:8766/`。页面默认显示当前训练的训练 loss，可切换查看验证历史。训练/验证条数根据数据集元数据和运行中的实际参数显示，包含命令行覆盖参数。远程 VS Code 连接服务器时，转发端口 `8766` 后在本机浏览器打开同一地址。网页每 5 秒读取状态，监控进程每 30 秒采样并保存训练数据。横轴从 0 显示到当前训练步；之前未记录的区间以灰色标明，不补造数据或跨缺口连线。要保存从开始起的采样历史，应在训练启动后同时运行监控。

也可以直接选择任意实验配置：

```bash
CUDA_VISIBLE_DEVICES=6,7 \
bash src/lerobot/policies/pi05/training/train.sh \
  src/lerobot/policies/pi05/training/configs/dualarm_fold_towel_rtc_frozen_vision.json \
  --print-command
```

确认后去掉 `--print-command` 才会启动。进程数由配置中的 `parallelism.dp_replicate * parallelism.dp_shard` 决定；两卡配置需要提供两张可用 GPU，`batch_size` 表示每卡 batch。

## 每次实验怎么保存配置

1. 从 `configs/` 复制一个 JSON，使用能识别实验的文件名；修改 `job_name`、`seed`、学习率或其他实验参数。
2. 为该次运行设置独立 `output_dir`。模板的 `output_dir=null` 使用 LeRobot 自动生成的 `outputs/train/日期/时间_job_name/`，在仓库根目录下；也可明确写入服务器 `/mnt/work/dakun/outputs/你的新实验目录`。
3. 用 `train.sh 你的配置.json --print-command` 检查，然后启动。
4. checkpoint 的 `pretrained_model/train_config.json` 保存**包含默认值和命令行覆盖的实际配置**，部署与恢复训练以这份配置为准。

因此，重复试验可以复用模板；改变实验设置时另存配置；恢复同一次训练使用 checkpoint 保存的配置。GPU 编号属于启动环境，保存配置时也应在实验记录里记下 GPU、代码提交和启动命令。

临时覆盖仍支持 LeRobot 原生参数，例如：

```bash
CUDA_VISIBLE_DEVICES=4 \
bash src/lerobot/policies/pi05/training/train.sh \
  src/lerobot/policies/pi05/training/configs/dualarm_fold_towel_rtc_lora.json \
  --print-command --seed=2000 --job_name=fold_towel_lora_seed2000
```

常用机器路径可用 `LEROBOT_PYTHON`、`DATASET_ROOT`、`MODEL_PATH`、`TOKENIZER_PATH`、`OUTPUT_DIR` 覆盖；显式 `--key=value` 优先于这些环境变量。正式实验优先把算法参数写入 JSON。

## 恢复已有训练

直接把 **checkpoint 的配置**作为第一个参数，加 `--resume=true`。不要用新训练模板替代它：

```bash
CUDA_VISIBLE_DEVICES=6,7 \
bash src/lerobot/policies/pi05/training/train.sh \
  /mnt/work/dakun/outputs/DualArm_FoldTowel_pi05_rtc_full_gpu5_6/checkpoints/015000/pretrained_model/train_config.json \
  --resume=true --print-command
```

将 step 替换为实际恢复点。原输出路径从该配置读取；同级 `training_state/` 必须完整，才能恢复优化器、调度器、随机状态等。新训练模板没有设置 `policy.use_peft=true`：LoRA 是在加载 base 后由训练器应用的；checkpoint 配置中的这个字段则记录已训练 adapter，不能盲目复制为新训练配置。

## 服务器路径与环境

| 内容      | 当前服务器路径                                  |
| --------- | ----------------------------------------------- |
| 代码      | `/mnt/work/dakun/lerobot`                       |
| Python    | `/mnt/work/dakun/conda-envs/lerobot/bin/python` |
| 数据      | `/mnt/work/dakun/DualArm_FoldTowel_joint_rtc`   |
| PI05 base | `/mnt/work/dakun/models/pi05_base`              |
| tokenizer | `/mnt/work/dakun/models/paligemma-3b-pt-224`    |

`LEROBOT_PYTHON` 可以显式指定 Python；否则依次使用已激活环境、仓库 `.venv`、工作目录下已有的 `conda-envs/lerobot`，最后使用 `python3`。配置检查只需要 Python 标准库。

现有训练环境继续使用即可。新建服务器环境时：

```bash
bash src/lerobot/policies/pi05/training/setup_server.sh
export LEROBOT_PYTHON="$PWD/.venv/bin/python"
```

安装使用 Python 3.12、Torch 2.7.1 + CUDA 11.8 和对应 TorchCodec，约束见 `server-constraints.txt`。本地 RTX 5090 部署使用 CUDA 12.8，安装入口为 [Flexiv 本地环境](../../../../../examples/flexiv/setup/setup_local_deploy.sh)。

原 `examples/flexiv/train_*.sh` 和 `setup_server.sh` 已移到此目录，历史训练输出和 checkpoint 路径保持原位置。
