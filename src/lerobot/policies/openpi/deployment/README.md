# OpenPI 双臂部署

本目录部署 `openpi/training` 训练的官方 OpenPI PyTorch π0.5 权重，默认使用
`openpi_pi05_dualarm_fold_towel_rtc_full/037500`。支持 RTC 和关闭 RTC 推理。
这是原始 OpenPI checkpoint，不需要转换为 LeRobot `pretrained_model/`。

## 运行

```bash
cd /home/nematrix/snap/lerobot/src/lerobot/policies/openpi/deployment

# 权重文件夹 → 步数 → 单步/连续；默认启用 RTC
bash run.sh

# 直接选 37,500 步，连续 RTC
bash run.sh 37500 --run-mode continuous

# 同一个 RTC 训练模型，关闭 RTC；每次推理执行前 25 个动作
bash run.sh 37500 --no-rtc --n-action-steps 25
# 等价的独立入口
bash run_no_rtc.sh 37500 --n-action-steps 25

# 仅高频记录状态，不保存图片；相机仍正常用于模型推理
bash run.sh 37500 --state-hz 200

# 记录 15 Hz 策略观测图像、状态和下发目标；可另加 --state-hz 200
bash run.sh 37500 --record
```

启动顺序：检查权重/配置 → 加载模型并离线预热 → 连接从臂和相机 →
完整路径预检、张开并确认夹爪到位、抬高回 home、关闭并确认夹爪到位 → 稳定 2 秒 →
等待 Enter → 开始模型控制。连续运行默认最多 60 秒，可用 `--duration-s` 调整。
`q` 在启动等待/单步菜单退出，连续运行用 Ctrl+C 停止。异常退出也先停止机械臂，再等待推理和日志写完。
模型控制只涉及两个从臂及夹爪，不连接主臂。

正常控制周期为 1/15 秒；某轮相机读取、推理处理或记录耗时导致超出计划周期时，
从该轮结束后重新等待一个周期，避免下一轮立即下发以追赶时间。
单步的超时检查覆盖推理后的相机/机械臂状态复查；超时不下发模型动作。

回 home 复用现有抬高路径：先到 `max(当前 TCP Z, home TCP Z + 0.10 m)`，
转移到 home 上方，再下降。启动与 `goHome.py` 调用同一个 `return_to_home` 函数：
目标使用 float64，避免 float32 的 `0.1 m` 舍入后被误判为夹爪越界。
部署启动时若双臂 14 个关节与 home 的最大误差不超过 `start_pose.joint_tolerance_rad`
（当前 0.02 rad，约 1.15°），跳过抬升回位并保持当前关节。夹爪已在关闭位置 1 mm 内则
不重发 home 动作，否则仅关闭夹爪。RTC 与无 RTC 均适用，任一关节超限仍走完整回位。
抬高归位不使用旧的起始关节差 1 rad 限制，仍在任何运动前完成关节限位、工具/TCP
一致性、整条 IK 路径与关节步长预检；策略动作的目标距离保护保持有效。
回位每点到位仍要求 TCP 位置误差不超过 3 mm；运动中的高度下限单独允许相邻规划点
最低 Z 以下 6 mm 的偏差，超出立即停止并报告左右臂、实测/规划高度及偏差。
不会自动清故障或初始化夹爪。

```bash
# 可单独回 home；只连接从臂，无模型和相机
.venv/bin/python goHome.py
# 夹爪初始化保持独立，执行时会运动夹爪
.venv/bin/python init_grippers.py
```

## 参数与记录

`deploy_config.json` 是独立配置；修改此文件不会更改 `pi05/deployment` 的配置。
硬件适配器、回 home 路径及记录器通过 `shared.py` 引用现有模块，避免两份硬件逻辑分叉。
这些公共模块仍需随仓库保留；本目录没有替换或修改它们。

- `rtc=true`：异步 RTC，50 个动作一块。按 LeRobot 的规则，以最近 100 次推理周转耗时的最大值
  估计前缀：`ceil(耗时 / (1/15))`，受可用旧动作数和上限约束；首次无旧块时前缀为 0。
  尚无稳定耗时样本时使用首次推理耗时；没有任何耗时估计但已有旧块时才使用上限。
  推理结束时间在工作线程记录，不把等待下一个 15 Hz 控制周期算作推理时间。
- `rtc_prefix_steps=8` 现在表示**前缀上限**，不是固定长度；不能超过权重训练的 `rtc_training_max_delay`。
  例如测得 115 ms 时通常固定 2 个动作，180 ms 时固定 3 个。每步去噪固定干净前缀、前缀 time=0。
  绝对目标前缀基于本次观测关节位置重新转换为相对量，夹爪仍是绝对宽度。
- 重新推理阈值为 `max(25/15 秒, 估计耗时 + 1/15 秒)`，慢推理可提前触发。
  新块需满足 `max(ceil(实际耗时×15), 推理期间已下发数量) <= 本次前缀长度`；
  通过后按两者较小值裁剪并替换未来队列，与 LeRobot 的裁剪规则一致。
  超出本次前缀覆盖范围时丢弃新块，继续旧队列并用更新后的耗时重新推理；连续 5 次丢弃或
  已开始运动后的 RTC 队列耗尽时抛错并停止机械臂。启动仍保持到首次推理完成。
  本地队列由主线程管理，LeRobot 在工作线程接受结果；接续规则对齐，不保证线程调度逐时刻相同。
- `--no-rtc`：没有前缀约束，执行 `n_action_steps` 个动作后保持并重新推理；不是以 RTC 队列方式接续。
- 单步：每按一次 Enter 重新推理，只执行第一个动作，等待到达，丢弃剩余动作；自动关闭 RTC。
- `max_joint_velocity_rad_s` / `max_joint_acceleration_rad_s2`：传给 RDK 的规划速度/加速度上限。
- `max_target_error_rad`：各关节目标相对实测位置的下发距离检查，不是到达容差。
- 高频状态记录使用独立采样线程和有界后台写队列；图片使用现有有界后台写进程。
  队列满时丢记录，不等待磁盘；CPU/GIL/磁盘仍可能影响性能，需要实机对比。

每次执行都创建 `/home/nematrix/snap/deployment_logs/openpi/<UTC时间>_<唯一标识>/`：

- `metadata.json`：实际 checkpoint、RTC 开关、步数及完整部署配置，即使未开记录也保存。
- `telemetry.jsonl`：下发的绝对目标、观测状态、相机年龄、推理/控制耗时、队列和前缀长度。
  `rtc_turnaround_estimate_ms`、`rtc_discarded_results`、`rtc_trimmed_actions` 分别记录耗时估计、
  累计丢弃结果数、最近一次接受结果的裁剪动作数；`prefix_length` 是最近请求的实际前缀长度。
- `robot_states/`：仅 `--state-hz` 启用时生成，含原始状态、时间戳、速度、差分加速度估计与采样摘要。
- 图片记录目录由 `--record-dir` 指定，默认在配置 telemetry 父目录的 `recordings/` 下。

## 环境与离线检查

本机已创建 `deployment/.venv`：复用 `lerobot-deploy` 的 CUDA PyTorch / RDK / RealSense，
只在隔离环境安装 transformers 4.53.2 及 OpenPI 依赖、替换官方 Gemma/SigLIP 模块。
`run.sh` 固定使用这个环境，不受终端当前 conda 环境影响。可用 `OPENPI_DEPLOY_PYTHON` 指定其他兼容环境。
不要在原有 LeRobot 环境降级 transformers，也不要在 OpenPI 根目录执行 `uv sync`。

迁移机器时：先准备已有的硬件部署环境，再执行：

```bash
LEROBOT_PYTHON=/path/to/lerobot-deploy/bin/python bash setup.sh
```

训练配置保留服务器路径，但部署只读取本机显式指定的权重、tokenizer、统计文件。
当前 tokenizer 复用 `pi05/checkpoints/DualArm_FoldTowel_pi05_rtc_full/tokenizer`；
加载时在内存中适配 transformers 5 保存的特殊 token 元数据，不改原文件。
归一化使用 checkpoint 自带 `dataset_stats.json`，相机顺序 image→base、image2→left wrist、image3→right wrist；
状态/动作顺序为左 7 关节+夹爪、右 7 关节+夹爪。14 关节相对本次观测，夹爪绝对值。

```bash
# 不加载模型，不连接任何硬件
bash run.sh --check
.venv/bin/python goHome.py --check
.venv/bin/python init_grippers.py --check

# 真模型离线推理，绝不连接硬件
bash run.sh --predict-npz /path/observation.npz --output /path/new_actions.npy
```

NPZ 需要 `observation.state`（16 维，rad/m）、三个
`observation.images.image{,2,3}`（224×224×3，RGB uint8，采集一致的 letterbox）。
输出为 50×16 的绝对关节/夹爪目标，输出路径必须不存在。
菜单只列出具有模型、训练配置、统计文件的数字步数目录，排除下载暂存目录。
加载使用 `strict=True`，不允许缺失权重后继续运行。

离线单元测试（不连接机器人）：

```bash
.venv/bin/python -m pytest test_deployment.py --confcutdir=. -q
```

环境安装脚本不强制安装测试工具；需要时在隔离环境安装 pytest。
验证结果见 `VALIDATION.md`。部署流程尚需操作者在实机上验证。
