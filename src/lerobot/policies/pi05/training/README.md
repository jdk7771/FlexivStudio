# π0.5 训练：配置定义实验，bash 负责启动

每次训练应能对应一份完整的最终配置。这里使用 LeRobot 原生 JSON 配置保存数据、模型、LoRA、学习率、batch、step 和验证参数；bash 只选择 Python 环境、GPU 和启动进程数。

```text
training/
├── configs/                           # 新训练的参数模板
│   ├── dualarm_fold_towel_rtc_lora.json
│   ├── dualarm_fold_towel_rtc_full.json
│   └── dualarm_fold_towel_rtc_lora_bs8.json
├── train.sh                           # 公共入口：JSON → 单进程或 torchrun
├── train_dualarm_fold_towel_rtc.sh     # LoRA / GPU 4 快捷入口
├── train_dualarm_fold_towel_rtc_full.sh # 全量 / GPU 6,7 快捷入口
├── train_dualarm_fold_towel_rtc_lora_bs8.sh # LoRA batch 8 / GPU 7
├── setup_server.sh                    # 服务器新建环境
└── server-constraints.txt
```

## 现有三组配置

配置按 [实验记录](../../../../../examples/flexiv/docs/EXPERIMENT_LOG.md) 中保存的 E1/E2/E3 `train_config.json` 提取。它们用于**新训练**，不包含旧训练的恢复状态。

| 模板                                   | 对应设置                               | 每卡 batch / 全局 batch | 学习率 | 默认 GPU（快捷脚本） |
| -------------------------------------- | -------------------------------------- | ----------------------- | ------ | -------------------- |
| `dualarm_fold_towel_rtc_lora.json`     | E1，动作专家和投影 LoRA，rank/alpha 32 | 16 / 16                 | `1e-4` | `4`                  |
| `dualarm_fold_towel_rtc_full.json`     | E2，冻结视觉编码器，其余全量微调       | 8 / 16                  | `1e-5` | `6,7`                |
| `dualarm_fold_towel_rtc_lora_bs8.json` | E3，同 E1，batch 8                     | 8 / 8                   | `1e-4` | `7`                  |

共同参数：BF16、梯度检查点、50,000 step、每 2,500 step 保存、每 1,000 step 验证最多 256 个样本。动作是关节相对当前观测、夹爪绝对值，chunk 50，training RTC 最大已知前缀 8。

数据为 `local/DualArm_FoldTowel_joint_rtc`，123 个 episode。`dataset.eval_split=0.08536585365853659` 与原转换清单按最后 10% 源 scene 留出验证集的结果一致：112 个训练 episode、11 个验证 episode。这是该数据集的固定划分；更换数据后应一起检查 `repo_id`、`root`、`eval_split` 和 `eval_steps`。

## 先看命令，再启动

从仓库根目录执行。`--print-command` 只读 JSON 并打印命令，不导入训练器、不读取训练数据或加载权重：

```bash
pi05_training=src/lerobot/policies/pi05/training
bash "$pi05_training/train_dualarm_fold_towel_rtc.sh" --print-command
bash "$pi05_training/train_dualarm_fold_towel_rtc_full.sh" --print-command
bash "$pi05_training/train_dualarm_fold_towel_rtc_lora_bs8.sh" --print-command
```

也可以直接选择任意实验配置：

```bash
CUDA_VISIBLE_DEVICES=6,7 \
bash src/lerobot/policies/pi05/training/train.sh \
  src/lerobot/policies/pi05/training/configs/dualarm_fold_towel_rtc_full.json \
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
