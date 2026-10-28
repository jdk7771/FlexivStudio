# 官方 openpi π0.5 + RTC：双臂叠毛巾

模型来自 Physical-Intelligence/openpi，官方源码提交固定为
`215abfb217dbac7d5f1273282331b9b1866c0479`。本目录使用官方
`openpi.models_pytorch.pi0_pytorch.PI0Pytorch`，在其上补 RTC 训练和 LeRobot v3 数据适配。

GitHub git HTTPS 在此机器超时，因此源码通过 GitHub 官方 codeload 归档下载；归档中的
commit 标记与 GitHub API 一致。来源和归档 SHA256 见 `UPSTREAM.json`，原始归档保存在
`/mnt/work/dakun/setup_artifacts/openpi-main.tar.gz`。这是官方源码快照，不含 git 历史。
两个上游文件的全部修改见 `rtc.patch`。
ALOHA/LIBERO 仿真子模块不用于这次真实机器人数据训练。

## 启动

```bash
cd /mnt/work/dakun/lerobot/src/lerobot/policies/openpi

# 只打印命令，不加载模型、不占 GPU
bash training/train_dualarm_fold_towel_rtc_full.sh --print-command

# CPU 检查：真实数据、视频解码、tokenizer、权重键名和形状
bash training/train_dualarm_fold_towel_rtc_full.sh --check-data

# 正式训练：由你手动执行
bash training/train_dualarm_fold_towel_rtc_full.sh
```

默认物理 GPU `0,1,2,3`，四个 DDP 进程，端口 `29520`。脚本可从任意工作目录执行。
`CUDA_VISIBLE_DEVICES`、`MASTER_PORT`、`OPENPI_PYTHON`、`OPENPI_CONFIG` 可覆盖启动环境；
实验参数写在 `configs/dualarm_fold_towel_rtc_full.json`。运行前不要 `uv sync`，该命令会换回
上游锁定的旧版 LeRobot 数据依赖，与本次 v3 数据适配不同。

## 与参考配置对齐

参考：`../../pi05/training/configs/dualarm_fold_towel_rtc_full.json`（相对本 training 目录）。
配置中保存参考文件的绝对路径和 SHA256，避免以后同名文件变化导致对照混淆。

| 项目 | 本次设置 |
| --- | --- |
| 训练方式 | 视觉编码器、VLM、动作专家全量训练；无 LoRA |
| 初始权重 | `/mnt/work/dakun/models/pi05_base/model.safetensors`，现有 openpi 格式权重 |
| 数据 | `/mnt/work/dakun/DualArm_FoldTowel_joint_rtc`，LeRobot v3 |
| 划分 | 同一 task 最后 ceil(123 × eval_split)=11 episodes 留出；112 train / 11 eval |
| 相机 | image → base，image2 → left_wrist，image3 → right_wrist |
| 动作 | 16 维；14 个关节相对当前 state，夹爪索引 7、15 保持绝对值 |
| 归一化/文本 | 相同 q01/q99 数据统计、16 维 state 文本、相同本地 PaliGemma tokenizer，200 tokens |
| action horizon | 50；输入 pad 到 32 维，loss 仅计算实际 16 维 |
| RTC | 每样本独立均匀采样 delay ∈ {0,…,8}；前缀为干净动作且 time=0；只计算后缀 loss |
| batch | 每卡 8 × 4 卡 = 32；梯度累积 1 |
| 精度 | 官方模型 BF16 参数策略 + bf16 autocast；梯度检查点 |
| 优化器 | AdamW，lr 1e-5，betas=(0.9,0.95)，eps=1e-8，weight_decay=1e-5，clip=1 |
| 调度器 | warmup 1000，cosine decay 50000，末端 2.5e-6；使用参考 LeRobot 的绝对 step 公式 |
| 步数/日志/保存 | 50000 / 每 50 步 / 每 2500 步 |
| 验证 | 每 1000 步，留出数据前 256 帧，batch 8 |
| 随机种子 | 1000；DDP 每 rank 使用 seed + rank |
| 图像增强/W&B/环境评测 | 关闭 |

这是“官方模型实现 + RTC”的对照实验，不是未修改的上游训练入口：

- `pi0_pytorch.py` 新增 `prefix_mask` 和逐动作时间编码，并允许关闭训练图像增强。
- 官方 transformers replacement 的 `GemmaRMSNorm` 增加逐动作 AdaRMS 条件支持。
- `training/train.py` 提供 v3 数据适配、与参考一致的 loss 归约/调度器、DDP、验证和保存。
- 官方模型内部精度策略仍与当前 LeRobot 移植版有差异；两者不能保证逐位一致。
- 验证由 rank 0 执行，其余 rank 等待；每次验证使用固定噪声种子并恢复训练 RNG，便于比较曲线。
  参考 LeRobot 会在所有 rank 上验证并使用持续变化的 RNG，因此验证 loss 不保证逐位一致。
- 本次补丁面向训练；上游 `sample_actions` 仍为普通采样，接实机 RTC 推理时需要额外接入硬前缀钳制。

## 输出与续训

输出目录：`/mnt/work/dakun/outputs/openpi_pi05_dualarm_fold_towel_rtc_full`。
非空目录不会被新训练覆盖。根目录保存实际配置、episode 划分、训练源码快照、
`train_loss.csv`、`eval_loss.csv`。每个 `checkpoints/002500` 等目录保存模型权重、优化器、
每 rank RNG、step/epoch/batch offset、配置和数据统计；先写 `.incomplete` 后原子改名。

```bash
bash training/train_dualarm_fold_towel_rtc_full.sh --resume \
  /mnt/work/dakun/outputs/openpi_pi05_dualarm_fold_towel_rtc_full/checkpoints/002500
```

续训自动使用检查点的 `train_config.json`，并恢复对应 epoch 的 batch 位置。
不要把 openpi 检查点直接交给 LeRobot 的 `lerobot-train --resume`。

### 从 1020 切到 1024 的 GPU 0、1、2、3 续训

两台机器通过 `/mnt/work/dakun` 访问共享文件，现有 `.venv` 在当前 1024 实例可用。
专用启动脚本固定使用物理 GPU `0,1,2,3`、默认端口 `29524`，自动选择输出目录中
编号最大的完整检查点，忽略 `.incomplete` 目录，并读取该检查点的原始配置。
恢复模型、AdamW 状态、绝对 step 对应的学习率、四个 rank 的 RNG 和数据 batch 位置；
每卡 batch 8，全局 batch 32，继续到总计 50000 步。

2026-10-27 检查时，最新完整检查点为 `007500`，日志最后记录为 7600；
这次从第 7500 步恢复，未保存的约 100 步需重新计算。原有 CSV 采用追加写入，
所以此前 7550、7600 的记录会保留，分析曲线时同 step 应取最后一条。

```bash
cd /mnt/work/dakun/lerobot/src/lerobot/policies/openpi

# 预览恢复路径和四卡启动命令，不启动训练
bash training/resume_dualarm_fold_towel_rtc_full_1024.sh --print-command

# 前台续训（由你执行）
bash training/resume_dualarm_fold_towel_rtc_full_1024.sh

# 或后台续训：与上面的前台命令二选一
nohup bash training/resume_dualarm_fold_towel_rtc_full_1024.sh \
  > /mnt/work/dakun/outputs/openpi_pi05_dualarm_fold_towel_rtc_full/resume_1024.log 2>&1 &
echo $!
tail -f /mnt/work/dakun/outputs/openpi_pi05_dualarm_fold_towel_rtc_full/resume_1024.log
```

也可以通过通用入口 `--resume latest` 自动选检查点，或 `--resume CHECKPOINT_DIR`
指定检查点。自动选择仅检查必要文件存在且非空，正式恢复仍由 PyTorch 校验内容。

## 环境

`.venv` 使用现有 `/mnt/work/dakun/conda-envs/lerobot` 作为只读基础，复用
PyTorch 2.7.1+cu118、torchcodec、FFmpeg 等。单独安装 transformers 4.53.2、
CPU JAX/Flax、Accelerate 1.10.1、LeRobot 0.4.4 的 v3 数据读取器，以及兼容现有 PyArrow
的 NumPy 2.2.6。当前本地 LeRobot 主线依赖 transformers 5 / huggingface_hub 1，不能直接与
官方 openpi 的 transformers 4.53.2 混用。这里只调用 0.4.4 的数据读取器，不调用它的策略。
官方 transformers patch 只写入本目录 `.venv`；不会覆盖基础环境。
这依赖现有基础环境继续保留；迁移机器时先准备对应基础环境，再执行：

```bash
bash training/setup.sh
```

`JAX_PLATFORMS=cpu` 避免官方模块导入时 JAX 占 GPU；实际训练完全用 PyTorch。
检查数据和单元测试不启动完整模型训练。全尺寸四卡显存/吞吐需要正式启动后确认。
