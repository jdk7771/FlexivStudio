# 本地 PI05 部署（RTC / 非 RTC）

本目录只负责模型选择、推理与机械臂执行；训练入口见 [training/README.md](../training/README.md)，
权重来源与存放方式见 [checkpoints/README.md](../checkpoints/README.md)。

## 入口索引

| 用途 | Bash | Python / 配置 |
| --- | --- | --- |
| RTC 模型部署 | `run.sh` | `scripts/deploy_dualarm.py` / `configs/deploy_config.json` |
| 非 RTC 同步推理 | `run_no_rtc.sh` | `scripts/deploy_dualarm_no_rtc.py` / `configs/deploy_config_no_rtc.json` |
| RTC 与高频状态记录 | `run_rtc_states.sh` | `scripts/deploy_rtc_states.py`，见 [高频记录说明](docs/rtc_states.md) |
| 回初始姿态 | — | `scripts/goHome.py` / `scripts/go_home_path.py` |
| 初始化夹爪 | — | `scripts/init_grippers.py` |
| 回放示教数据 | — | `scripts/replay_data.py` |

`scripts/flexiv_deploy_contract.py` 负责文件与配置检查，`scripts/flexiv_deploy_hardware.py` 负责设备适配，
`scripts/deployment_recording.py` 与 `scripts/high_rate_state_recording.py` 负责记录；`tests/test_*.py` 是离线或模拟设备检查。

## 历史模型与本次新模型

| 权重分组 | 对应训练配置 | 来源与关键设置 |
| --- | --- | --- |
| `DualArm_FoldTowel_pi05_rtc_full` | `dualarm_fold_towel_rtc_full.json` | 历史服务器 `rtc_full`；四卡 × batch 8，BF16，lr `1e-5`，开启验证 |
| `DualArm_FoldTowel_pi05_rtc_full_finetune` | `dualarm_fold_towel_rtc_full_finetune.json` | 本次 GPU `6,7`；双卡 × batch 16，FP32 / BF16，lr `2.5e-5`，无验证 |
| `DualArm_FoldTowel_pi05_rtc_frozen_vision` | `dualarm_fold_towel_rtc_frozen_vision.json` | 历史目录名含 full，实际冻结视觉编码器 |
| `DualArm_FoldTowel_pi05_no_rtc_full` | `dualarm_fold_towel_no_rtc_full.json` | 独立无 RTC 实验，历史保存配置优先 |

机器人机器上的 `/home/nematrix/snap/lerobot/src/lerobot/policies/pi05/checkpoints/DualArm_FoldTowel_pi05_rtc_full`
按既有下载记录对应历史 `rtc_full`，来源与校验值见 [checkpoint 说明](../checkpoints/README.md)。
其实际参数由 checkpoint 内的 `train_config.json` 决定；新训练只写入独立的 `rtc_full_finetune` 分组。

## 本次 RTC 全量微调模型

新增训练组为 `DualArm_FoldTowel_pi05_rtc_full_finetune`，默认使用 GPU `6,7` 训练，
50,000 steps、不做测评，计划保留 `010000`、`020000`、`030000`、`040000`、`050000`。
当前只准备了训练入口，尚无本次训练权重。

训练输出直接保存在服务器的
`/mnt/data/dakun/workspace/lerobot/src/lerobot/policies/pi05/checkpoints/DualArm_FoldTowel_pi05_rtc_full_finetune/checkpoints/`。
模型目录菜单和快捷脚本均支持 `<实验名>/checkpoints/<step>/pretrained_model/` 训练布局，
也支持原有的 `<实验名>/<step>/pretrained_model/` 下载布局。
在机器人机器使用时，可保留这两种目录结构之一。
训练使用的 PaliGemma tokenizer 也需在机器人机器就绪；可放入 `pretrained_model/tokenizer/`，
此时入口会自动采用它。没有内置 tokenizer 时，用 `--tokenizer /本机/tokenizer目录` 显式指定。

权重和 tokenizer 就绪后，在机器人机器的仓库根目录执行：

```bash
# 只检查模型文件，不加载模型或连接设备
bash src/lerobot/policies/pi05/deployment/run.sh DualArm_FoldTowel_pi05_rtc_full_finetune 050000 --check

# 使用本次 50k 权重进行 RTC 单步部署
bash src/lerobot/policies/pi05/deployment/run.sh DualArm_FoldTowel_pi05_rtc_full_finetune 050000 --run-mode step
```

新训练沿用 `chunk_size=50`、`rtc_training_max_delay=8`，对应现有 RTC 部署参数。
显式选择 `rtc_full_finetune` 分组即可复用现有部署入口。`configs/deploy_config.json` 中的默认组仍是历史
`DualArm_FoldTowel_pi05_rtc_full`；其机器人、相机和起点路径属于机器人机器。

## 原有部署入口与历史模型

这是 `examples/flexiv/` 部署代码的一份副本，和 [本地 checkpoints](../checkpoints/README.md) 放在同一 `pi05/` 目录下。来源代码提交为 `ab265aae`；复制了部署入口、checkpoint 检查、RDK/RealSense 适配、回初始姿态入口及离线测试。

以下为原机器人机器的部署记录，具体已下载步数以该机器文件为准。旧实验 `DualArm_FoldTowel_pi05_rtc_full_gpu5_6` 实际冻结视觉编码器，本地已从 `rtc_full` 改名为 `DualArm_FoldTowel_pi05_rtc_frozen_vision`；原下载记录包含 25k/30k/40k/45k/50k。历史全量训练来自 `101.132.150.51:1024` 的 `/mnt/work/dakun/logs/projects/lerobot/rtc_full`，15k/20k/25k 使用本地组 `DualArm_FoldTowel_pi05_rtc_full`，保存配置均确认视觉编码器参与训练。RTC 默认选择此历史组的 `last`，原记录指向 25k；冻结组使用 `DualArm_FoldTowel_pi05_rtc_frozen_vision` 显式选择。两组不能按相同步数混放。菜单中的 `full` 表示完整模型文件格式，视觉是否训练另外从保存的 `config.json` 显示。

模型菜单先列出 `checkpoints/` 下含有权重的实验文件夹，再列出所选文件夹中的 `<step>/pretrained_model/` 或 `checkpoints/<step>/pretrained_model/`，按训练步数从大到小排列。文件夹和步数根据本地文件自动发现；隐藏下载目录、没有模型配置或权重的目录不会列入，`last` 链接不重复显示。原 `examples/flexiv/` 入口仍可单独使用。以后修改两份部署代码时应核对差异。

RTC 与非 RTC 使用独立部署入口、模型加载和执行循环：`run.sh` → `scripts/deploy_dualarm.py`，使用 `configs/deploy_config.json`，固定 trained RTC；`run_no_rtc.sh` → `scripts/deploy_dualarm_no_rtc.py`，使用 `configs/deploy_config_no_rtc.json`，固定同步动作块推理。RTC 入口要求兼容的 RTC 训练权重；非 RTC 入口支持普通权重以及 RTC 训练权重，并在运行时设置 `rtc_config=None`，不会修改 checkpoint 中的训练配置或权重。非 RTC 配置默认指向 `DualArm_FoldTowel_pi05_no_rtc_full/025000`。两套共用设备适配层和基础文件检查。

非 RTC 连续模式在队列为空时保持当前位置、从最新观测生成一个动作块，再按 15 Hz 逐个执行 `n_action_steps` 个动作（部署配置默认 50）。同一块的关节增量始终使用生成该块时的姿态作基准，夹爪保持绝对量；执行完指定步数后丢弃该块其余动作并再次推理，因此块之间会有推理停顿。推理结束前重新检查相机与从臂状态，观测过期或推理超时即停止，不追赶补发。单步模式每次重新推理、只发送一个动作并丢弃剩余动作。两种推理方式均保留回位、按键启动和目标限位/跟随偏差检查。

```text
deployment/
├── README.md
├── run.sh                       # RTC
├── run_no_rtc.sh                # 非 RTC
├── run_rtc_states.sh            # RTC 与高频状态记录
├── scripts/                     # Python 部署入口、设备/回位/记录模块和工具
├── configs/                     # RTC 与非 RTC 部署 JSON
├── tests/                       # 六组离线/模拟设备测试
└── docs/
    └── rtc_states.md            # 高频状态记录说明
```

训练侧新增本次 `rtc_full_finetune`，并保留 `rtc_frozen_vision`、`rtc_full`、`no_rtc_full` 三组，与这里的完整 checkpoint 组名一一对应；
openpi joint 和 pose 使用各自独立入口。新训练名称与历史服务器目录的映射见
[实验索引](../LOCAL_WORKFLOW.md) 和 [训练配置说明](../training/README.md)。
保存配置和历史目录不因名称整理而改写。

## 检查本地 checkpoint

在仓库根目录运行，`--check` 只读取模型文件，不加载模型，也不连接硬件：

```bash
deployment=src/lerobot/policies/pi05/deployment
bash "$deployment/run.sh" 015000 --check
bash "$deployment/run.sh" 020000 --check
bash "$deployment/run.sh" 025000 --check
bash "$deployment/run.sh" DualArm_FoldTowel_pi05_rtc_frozen_vision 050000 --check
bash "$deployment/run.sh" DualArm_FoldTowel_pi05_rtc_frozen_vision 045000 --check
bash "$deployment/run.sh" DualArm_FoldTowel_pi05_rtc_frozen_vision 040000 --check
bash "$deployment/run.sh" DualArm_FoldTowel_pi05_rtc_frozen_vision 030000 --check
bash "$deployment/run_no_rtc.sh" 025000 --check
```

从其他工作目录也可以使用 `run.sh` 的绝对路径。`LEROBOT_PYTHON` 可指定 Python；默认使用已激活环境，否则使用 `/home/nematrix/snap/.conda-envs/lerobot-deploy/bin/python`。

## 部署

按希望使用的推理方式选择启动脚本。不带参数会依次选择权重文件夹、checkpoint 步数、单步或连续运行；两个权重菜单均可输入 `q` 退出：

```bash
bash src/lerobot/policies/pi05/deployment/run.sh          # RTC
bash src/lerobot/policies/pi05/deployment/run_no_rtc.sh   # 非 RTC
```

也可直接指定文件夹和步数，跳过两个权重菜单：

```bash
bash src/lerobot/policies/pi05/deployment/run.sh DualArm_FoldTowel_pi05_rtc_full 025000 --run-mode step
bash src/lerobot/policies/pi05/deployment/run.sh DualArm_FoldTowel_pi05_rtc_frozen_vision 050000 --run-mode step
bash src/lerobot/policies/pi05/deployment/run_no_rtc.sh DualArm_FoldTowel_pi05_no_rtc_full 025000 --run-mode step
```

同一份 RTC 50k 权重关闭 RTC 做对照，在本目录运行：

```bash
bash run_no_rtc.sh DualArm_FoldTowel_pi05_rtc_frozen_vision 050000 --check  # 只检查文件
bash run_no_rtc.sh DualArm_FoldTowel_pi05_rtc_frozen_vision 050000         # 同步推理，再选择单步/连续
```

这仍然加载 RTC 训练的 50k 参数，仅关闭推理时的 RTC 前缀衔接和异步调度；默认每块执行 50 个动作。日志仍写入非 RTC 配置的 `telemetry` 路径。

只写步数时，RTC 脚本默认选择 `DualArm_FoldTowel_pi05_rtc_full`，非 RTC 脚本默认选择 `DualArm_FoldTowel_pi05_no_rtc_full`：

```bash
bash src/lerobot/policies/pi05/deployment/run.sh 025000 --run-mode step
bash src/lerobot/policies/pi05/deployment/run_no_rtc.sh 025000 --run-mode step
```

已激活部署环境时，也可以在本目录直接运行 `python scripts/deploy_dualarm.py`（RTC）或 `python scripts/deploy_dualarm_no_rtc.py`（非 RTC）。

非 RTC 每次推理后执行多少步，可在 `configs/deploy_config_no_rtc.json` 设置 `"n_action_steps": 10`，或启动时用 `--n-action-steps 10` 覆盖。合法范围为 `1..chunk_size`，当前为 1–50；默认 50。`chunk_size=50` 决定模型预测长度，`num_inference_steps=10` 决定去噪次数。单步模式始终每按 Enter 执行一个动作。

```bash
bash src/lerobot/policies/pi05/deployment/run_no_rtc.sh 025000 --n-action-steps 10 --run-mode continuous
bash src/lerobot/policies/pi05/deployment/run_no_rtc.sh 025000 --n-action-steps 10 --check
```

新全量 25k 同样可以通过 `bash run_no_rtc.sh DualArm_FoldTowel_pi05_rtc_full 025000 --check` 检查，并通过 `bash run_no_rtc.sh DualArm_FoldTowel_pi05_rtc_full 025000` 运行普通同步推理。本次三份权重均通过实际 GPU 离线 RTC 推理，25k 另通过非 RTC 离线推理；预测结果和运行日志保存在 `../checkpoints/DualArm_FoldTowel_pi05_rtc_full/validation/`。

任意本地权重路径也可通过 `--checkpoint /绝对路径/pretrained_model` 指定。`--check` 保留非交互行为：未指定权重时检查配置中的默认 checkpoint。

上述部署命令会连接本机机械臂和相机，自动回到 home（采集数据的首帧姿态），到位后等待按键。连续模式按 Enter 才开始执行策略，输入 `q` 退出；单步模式每按一次 Enter 执行一个动作。连续运行使用 `--run-mode continuous`，默认 60 秒从按键后的就绪检查完成时开始计时。即使通过参数指定模型和运行方式，仍须在回位后按 Enter 才执行策略。

设备序列号、网络接口、力矩限制和初始姿态来源沿用现有 Flexiv 部署配置，详见 [原部署说明](../../../../../examples/flexiv/docs/README_local_deploy.md)。

连接时对工具列表和当前工具的只读查询遇到 `No reply from the robot` 会在默认 15 秒窗口内间隔重试；超时、断连或故障中止。已选中的工具直接沿用，使能后不再重复查询或切换。此重试由模型部署、`scripts/goHome.py` 和数据回放共用，不重复发送使能、工具切换或运动指令。

`configs/deploy_config.json` 指向新 RTC 全量训练的 25k 权重及其 tokenizer；`configs/deploy_config_no_rtc.json` 指向非 RTC 25,000 步权重及其内置 tokenizer。全量 checkpoint 无需加载 base 权重。实际训练配置、预处理文件中的服务器路径保留原始内容，模型与 tokenizer 由各自部署入口显式指定本地路径。

两套入口均可用 `--config /绝对路径/配置.json` 指定设备配置。观测首帧仍从本机 `/home/nematrix/snap/reports/dual_arm_fold_towel_2026-10-23/conversion_manifest.json` 与其指向的采集 HDF5 读取；默认日志分别写入 `/home/nematrix/snap/deployment_logs/DualArm_FoldTowel_rtc_full.jsonl` 和 `/home/nematrix/snap/deployment_logs/DualArm_FoldTowel_no_rtc_full.jsonl`。

## RTC 参数与更新节奏

当前 `chunk_size=50` 表示每次预测 50 步，`rtc_queue_threshold=25` 表示剩余队列降到 25 步时后台开始下一次推理。新结果返回时，根据推理期间实际消耗的旧动作对齐并替换未来队列；因此稳定运行通常约每 25 步更新一次，不保证每块执行完 50 步。

`training_max_delay=8` 必须匹配 checkpoint 的 `rtc_training_max_delay`：训练为每个样本随机使用 0–8 步干净动作前缀，部署的前缀条件和实际推理延迟必须处于这一范围。`rtc_execution_horizon=10` 在此部署中是提供给模型的旧动作前缀容量，不是每次执行步数；trained 模式实际固定的前缀长度由延迟估计决定，并受训练上限 8 约束。超过所条件化前缀的结果会被丢弃，连续失败或队列耗尽会停止。15 Hz 下 8 步对应约 0.53 秒，和两次推理更新之间约 1.67 秒是不同的时间段。

## 可选记录图像和机械臂数据

两个部署入口均支持 `--record`，默认关闭；显式关闭用 `--no-record`。在本目录运行：

```bash
# RTC 50k，连续运行并记录
bash run.sh DualArm_FoldTowel_pi05_rtc_frozen_vision 050000 --run-mode continuous --record

# 同一份 RTC 权重关闭 RTC，每批执行 25 步，并记录
bash run_no_rtc.sh DualArm_FoldTowel_pi05_rtc_frozen_vision 050000 --n-action-steps 25 --run-mode continuous --record

# 指定记录父目录，每次运行自动创建独立子目录
bash run.sh DualArm_FoldTowel_pi05_rtc_frozen_vision 050000 --record --record-dir /home/nematrix/snap/deployment_logs/recordings
```

仍然先回 home、等待 Enter，再开始策略和记录。只记录策略运行阶段，包含 RTC 首批动作等待时的观测；不记录加载模型、回 home 或等待按键过程。单步模式每个成功执行并到位的动作保存一条。`--check --record` 不创建记录进程、文件或连接设备。

默认保存到配置中 `telemetry` 同级的 `recordings/<UTC时间>_<rtc或no_rtc>_<随机标识>/`：

- `metadata.json`：实际部署配置、checkpoint 路径、RTC 开关、关节顺序、单位和运行方式。
- `frames.jsonl`：每条记录的序号、图像路径、相机接收时间、机械臂状态读取时间，以及原 telemetry 数据（双臂共 14 个关节角、2 个夹爪宽度、实际下发的绝对目标、队列长度等）。关节单位 rad，夹爪单位 m；连续模式保持期间目标为 `null`。
- `images/image/`、`images/image2/`、`images/image3/`：三路 **224×224、补边后的模型输入 RGB**，JPEG quality=90；不是 1280×720 原始视频，也不包含深度图。
- `summary.json`：提交/写入数量、丢弃数量、未写完数量、写盘错误，以及主线程提交记录的平均/最大耗时。

图像使用固定 8 个共享内存槽（约 3.45 MiB）。独立低优先级进程负责 JPEG 编码、JSON 序列化和写盘，不加载模型、不使用 CUDA、不另开相机；控制循环只复制现成的小图并非阻塞入队。缓存满时丢弃整条记录（图像与状态一起丢弃），序号保留缺口；写盘失败仅停止记录，部署继续，终端和汇总会提示。启用记录时原 telemetry 文件也由后台进程写入，因此溢出时同样会缺少对应行；关闭记录保持原 telemetry 行为。

记录的是每个控制周期读取的观测与该周期下发的目标。RTC 异步推理可能使用更早观测，不能把同一行图像直接当成该动作的推理源图像；这里的时间戳是主机接收/读取时间，不是硬件严格同步时间。非 RTC 推理后的相机复查不会覆盖这一条记录的原始观测时间。

正常结束、异常或 Ctrl+C 时先停止机械臂和推理，再限时排空写盘队列；强制杀进程/断电可能留下部分图像或缺少汇总。记录过程不会等待磁盘，但图像复制、进程调度和系统 I/O 仍有开销，不能保证严格零性能影响；使用 `summary.json` 的提交耗时和部署控制周期日志评估实际机器负载。需要完整不丢帧的高清采集时，应单独规划采集带宽。

## goHome 先抬升再回位

本目录的 `scripts/goHome.py`、RTC 部署和无 RTC 部署共用同一回位流程：预检整条路径 → 打开双夹爪 → 两臂分别沿基座/WORLD 的 +Z 抬升 → 在高处移到各自目标正上方并调整姿态 → 固定目标 XY 和姿态下降 → 到位后关闭夹爪。回位不调用 `Init()`。部署须完成回位和夹爪关闭后，才进入原有的静置与策略启动流程。

模型部署启动时先比较双臂 14 个关节与 home 目标：最大误差不超过 `start_pose.joint_tolerance_rad`（当前 0.02 rad，约 1.15°）就跳过抬升回位，保持当前关节位置。此时若双夹爪已在关闭位置 1 mm 内，不重发 home 动作；否则仅保持当前关节并关闭夹爪。任一关节超过容差仍执行完整路径。OpenPI 的 RTC/无 RTC 部署同样适用。单独执行 `scripts/goHome.py` 仍执行完整回位流程。

这三个入口的回位不再受 `start_pose.max_joint_delta_rad`（原先 1 rad 起始距离）限制，仍检查关节范围、整条 IK 路径、逐点步长和到位误差。两种部署默认使用与 goHome 相同的 0.10 m 高度余量，由 `scripts/go_home_path.py` 的 `DEFAULT_LIFT_M` 设置。

夹爪初始化由独立脚本 `scripts/init_grippers.py` 执行：连接并使能左右从臂、保持当前关节位置，两侧夹爪各调用一次 `Init()`，不执行 home 路径、不连接相机、不读取采集数据。RDK 1.9.2 的 `Init()` 返回仅代表请求送达，没有初始化完成标志，因此先等待至少 10 秒，再要求两侧夹指连续 0.5 秒停止运动；总等待上限 30 秒。初始化请求失败、机械臂故障或等待超时均中止，不自动重发 `Init()`。

```bash
python scripts/init_grippers.py --check   # 只检查配置，不连接设备
python scripts/init_grippers.py           # 单独初始化两侧夹爪，不自动回 home
```

每侧经过高度为 `max(当前 TCP Z, 采集首帧 TCP Z + lift_m)`，默认以最终 home 高度为基准加 0.10 m。当前低于此高度则先抬到此高度；当前已更高则保持当前高度平移，到目标正上方后再下降，不再额外向上抬 10 cm。两臂都完成一段才进入下一段。goHome 的名义 TCP 路径速度为 0.10 m/s（10 cm/s），实际速度受逐点到位等待及关节速度/加速度限制约束。此值在 `scripts/go_home_path.py` 的 `TCP_SPEED_M_S` 中设置。

```bash
python scripts/goHome.py --check          # 读取目标和抬升参数，不连接硬件
python scripts/goHome.py                  # 以 home 高度 + 10 cm 回位，当前更高则保持高度平移
python scripts/goHome.py --lift-m 0.15    # 高度余量改为 15 cm，允许 2–30 cm
```

目标 TCP 与目标关节从采集首帧同一时间戳读取。移动前用 RDK `Model.reachable(..., free_orientation=False)` 预检两臂整条路径；工具标定不匹配、不可达、关节越界或 IK 跳变均中止，不回退原关节插值路径。TCP 路径以不超过 5 mm / 0.03 rad 的点离散为关节目标，每段关节变化不超过原配置的 0.05 rad；下降时逐渐恢复采集首帧的冗余关节姿态。运行中同时检查关节与 TCP 到位。

到位误差与运动中高度下限分别设置：每点仍须达到 3 mm 位置误差内才接受；运动中允许低于相邻两个理想路径点的最低 Z 最多 6 mm（`HEIGHT_CORRIDOR_TOLERANCE_M`），为上一点的到位误差和后续瞬态偏差各留 3 mm。此下限始终依据理想路径，不随实测高度累计下移。超限仍立即停止，并输出路径点、左右臂、实际/规划/最低允许 Z 和向下偏差。OpenPI 部署与其 goHome 也复用此模块。

如果某个 TCP 小步的 IK 结果仍超过关节步长限制，预检会对该段 TCP 路径二分细化，并以上一个已通过的关节解重新求解，每个原始段最多细分 8 层。细分后仍超限则在移动前退出，报错列出阶段、左右臂、关节编号、前后角度及变化量；整条路径最多接受 4096 个点。细分不会放宽 `start_pose.waypoint_step_rad`。

每个路径点须同时满足 TCP 位置误差不超过 3 mm、姿态误差不超过 0.03 rad、关节误差不超过配置中的 0.02 rad。单点等待上限 5 秒，总回程上限沿用 `start_pose.timeout_s`。等待超过 1 秒后，每秒输出路径点编号、阶段进度、左右臂未满足项（`position` / `orientation` / `joints`）、各项误差以及实际/目标 TCP XYZ；超时前也会输出完整诊断。超时会停止，不跳过该点。

这是小步 IK 生成的 TCP 路径，段间仍由关节控制器插值；没有整臂、工具或双臂碰撞规划，也未标定桌面高度。10 cm 是路径高度余量，不是已验证的桌面安全距离。数据回放以及 `examples/flexiv/goHome.py` 保留各自原有回位路径。

接口依据：[RDK 1.9.2 Model（只读 IK 可达性检查）](https://github.com/flexivrobotics/flexiv_rdk/blob/v1.9.2/include/flexiv/rdk/model.hpp)、[RobotStates TCP 位姿格式](https://www.flexiv.com/software/rdk/api/structflexiv_1_1rdk_1_1_robot_states.html)。

## 回放一条示教数据

在部署环境中运行，不带参数列出本地 50 条示教，输入编号或 `scene_0001` 这样的名称选择：

```bash
cd /home/nematrix/snap/lerobot/src/lerobot/policies/pi05/deployment
python scripts/replay_data.py
```

程序读取转换清单指向的原始 `scene_*/lowdim/lowdim.h5`，每条 scene 是一次完整采集，包含训练过滤前的轨迹，不按训练 episode 的切分截断。回放的是从臂实测关节位置和记录的夹爪指令；原数据未保存关节控制指令。仅连接配置中的左右从臂及夹爪，不连接主臂、相机，不加载 checkpoint。

选定后先检查数据与从臂序列号、时间戳、采样间隙、单位及关节限位，再自动分段对齐**所选数据的第一帧**。到位后按 Enter 才开始回放，输入 `q` 退出；运行中按 Ctrl+C 停止两台从臂。仍保留起点最大偏差 1 rad 和运行目标相对实测值最大偏差 0.25 rad，超限即停止，不跳过这些检查。请先断开采集遥操作或其他控制程序。

```bash
python scripts/replay_data.py --list                              # 仅列出数据
python scripts/replay_data.py --scene scene_0001 --check           # 仅检查文件，不连接设备
python scripts/replay_data.py --scene scene_0001                  # 指定数据，回位后仍等 Enter
python scripts/replay_data.py --scene scene_0001 --speed 0.5      # 半速，时间约加倍
```

默认原速 `--speed 1`，支持 `(0, 1]`。关节按原始时间线插值到 15 Hz，夹爪取当前采样时刻之前最后一条开合指令，末帧也会发送并等待到位。SDK 调用迟到时顺延，保留全部重采样帧，不追赶或跳帧；速度/加速度限制可能延长实际时长，跟随偏差过大会中止。`--check` 检查本地数据，机器人关节限位及实际跟随能力需连接后检查。

## 离线测试

```bash
PYTHONPATH="src:src/lerobot/policies/pi05/deployment/scripts${PYTHONPATH:+:$PYTHONPATH}" \
UV_CACHE_DIR=/tmp/flexivstudio-uv \
/home/nematrix/snap/.conda-envs/lerobot-deploy/bin/uv run --no-project \
  --python /home/nematrix/snap/.conda-envs/lerobot-deploy/bin/python \
  python -m unittest discover -s src/lerobot/policies/pi05/deployment/tests -p 'test_*.py' -v
```
