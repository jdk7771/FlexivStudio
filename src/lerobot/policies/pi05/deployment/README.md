# 本地 PI05 部署（RTC / 非 RTC）

这是 `examples/flexiv/` 部署代码的一份副本，和 [本地 checkpoints](../checkpoints/README.md) 放在同一 `pi05/` 目录下。来源代码提交为 `ab265aae`；复制了部署入口、checkpoint 检查、RDK/RealSense 适配、回初始姿态入口及离线测试。

模型菜单先列出 `checkpoints/` 下含有权重的实验文件夹，再列出所选文件夹中的 `<step>/pretrained_model/`，按训练步数从大到小排列。文件夹和步数根据本地文件自动发现；隐藏下载目录、没有模型配置或权重的目录不会列入，`last` 链接不重复显示。原 `examples/flexiv/` 入口仍可单独使用。以后修改两份部署代码时应核对差异。

RTC 与非 RTC 使用独立部署入口、模型加载和执行循环：`run.sh` → `deploy_dualarm.py`，使用 `deploy_config.json`，固定 trained RTC；`run_no_rtc.sh` → `deploy_dualarm_no_rtc.py`，使用 `deploy_config_no_rtc.json`，固定同步动作块推理。RTC 入口要求兼容的 RTC 训练权重；非 RTC 入口支持普通权重以及 RTC 训练权重，并在运行时设置 `rtc_config=None`，不会修改 checkpoint 中的训练配置或权重。非 RTC 配置默认指向 `DualArm_FoldTowel_pi05_no_rtc_full_gpu7_6_5_4/025000`。两套共用设备适配层和基础文件检查。

非 RTC 连续模式在队列为空时保持当前位置、从最新观测生成一个动作块，再按 15 Hz 逐个执行 `n_action_steps` 个动作（部署配置默认 50）。同一块的关节增量始终使用生成该块时的姿态作基准，夹爪保持绝对量；执行完指定步数后丢弃该块其余动作并再次推理，因此块之间会有推理停顿。推理结束前重新检查相机与从臂状态，观测过期或推理超时即停止，不追赶补发。单步模式每次重新推理、只发送一个动作并丢弃剩余动作。两种推理方式均保留回位、按键启动和目标限位/跟随偏差检查。

```text
deployment/
├── run.sh                       # RTC 启动脚本
├── deploy_dualarm.py             # RTC 部署入口、单步/连续运行
├── deploy_config.json            # RTC 配置，默认 rtc_full/last
├── run_no_rtc.sh                 # 非 RTC 启动脚本
├── deploy_dualarm_no_rtc.py       # 非 RTC 独立部署入口、单步/连续运行
├── deploy_config_no_rtc.json      # 非 RTC 配置，默认 gpu7_6_5_4/025000
├── flexiv_deploy_contract.py     # 离线配置、权重、统计检查
├── flexiv_deploy_hardware.py     # RDK / RealSense 适配
├── goHome.py                     # 回到采集首帧姿态
├── init_grippers.py              # 独立初始化两侧夹爪
├── go_home_path.py              # goHome 抬升、上方平移、下降路径与 IK 预检
├── replay_data.py                # 选择一条原始示教，仅回放左右从臂
├── test_local_deploy.py          # 离线 / 模拟设备测试
├── test_non_rtc_deploy.py         # 非 RTC 单步、连续、超时与起点等待测试
└── test_replay_data.py           # 数据回放与模拟时钟测试
```

## 检查本地 checkpoint

在仓库根目录运行，`--check` 只读取模型文件，不加载模型，也不连接硬件：

```bash
deployment=src/lerobot/policies/pi05/deployment
bash "$deployment/run.sh" 050000 --check
bash "$deployment/run.sh" 045000 --check
bash "$deployment/run.sh" 040000 --check
bash "$deployment/run.sh" 030000 --check
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
bash src/lerobot/policies/pi05/deployment/run.sh rtc_full 050000 --run-mode step
bash src/lerobot/policies/pi05/deployment/run_no_rtc.sh DualArm_FoldTowel_pi05_no_rtc_full_gpu7_6_5_4 025000 --run-mode step
```

同一份 RTC 50k 权重关闭 RTC 做对照，在本目录运行：

```bash
bash run_no_rtc.sh rtc_full 050000 --check  # 只检查文件
bash run_no_rtc.sh rtc_full 050000         # 同步推理，再选择单步/连续
```

这仍然加载 RTC 训练的 50k 参数，仅关闭推理时的 RTC 前缀衔接和异步调度；默认每块执行 50 个动作。日志仍写入非 RTC 配置的 `telemetry` 路径。

只写步数时，RTC 脚本默认选择 `rtc_full`，非 RTC 脚本默认选择 `DualArm_FoldTowel_pi05_no_rtc_full_gpu7_6_5_4`：

```bash
bash src/lerobot/policies/pi05/deployment/run.sh 050000 --run-mode step
bash src/lerobot/policies/pi05/deployment/run_no_rtc.sh 025000 --run-mode step
```

已激活部署环境时，也可以在本目录直接运行 `python deploy_dualarm.py`（RTC）或 `python deploy_dualarm_no_rtc.py`（非 RTC）。

非 RTC 每次推理后执行多少步，可在 `deploy_config_no_rtc.json` 设置 `"n_action_steps": 10`，或启动时用 `--n-action-steps 10` 覆盖。合法范围为 `1..chunk_size`，当前为 1–50；默认 50。`chunk_size=50` 决定模型预测长度，`num_inference_steps=10` 决定去噪次数。单步模式始终每按 Enter 执行一个动作。

```bash
bash src/lerobot/policies/pi05/deployment/run_no_rtc.sh 025000 --n-action-steps 10 --run-mode continuous
bash src/lerobot/policies/pi05/deployment/run_no_rtc.sh 025000 --n-action-steps 10 --check
```

任意本地权重路径也可通过 `--checkpoint /绝对路径/pretrained_model` 指定。`--check` 保留非交互行为：未指定权重时检查配置中的默认 checkpoint。

上述部署命令会连接本机机械臂和相机，自动回到 home（采集数据的首帧姿态），到位后等待按键。连续模式按 Enter 才开始执行策略，输入 `q` 退出；单步模式每按一次 Enter 执行一个动作。连续运行使用 `--run-mode continuous`，默认 60 秒从按键后的就绪检查完成时开始计时。即使通过参数指定模型和运行方式，仍须在回位后按 Enter 才执行策略。

设备序列号、网络接口、力矩限制和初始姿态来源沿用现有 Flexiv 部署配置，详见 [原部署说明](../../../../../examples/flexiv/docs/README_local_deploy.md)。

连接时对工具列表和当前工具的只读查询遇到 `No reply from the robot` 会在默认 15 秒窗口内间隔重试；超时、断连或故障中止。已选中的工具直接沿用，使能后不再重复查询或切换。此重试由模型部署、`goHome.py` 和数据回放共用，不重复发送使能、工具切换或运动指令。

`deploy_config.json` 指向 RTC 最终权重和共享 tokenizer；`deploy_config_no_rtc.json` 指向非 RTC 25,000 步权重及其内置 tokenizer。全量 checkpoint 无需加载 base 权重。实际训练配置、预处理文件中的服务器路径保留原始内容，模型与 tokenizer 由各自部署入口显式指定本地路径。

两套入口均可用 `--config /绝对路径/配置.json` 指定设备配置。观测首帧仍从本机 `/home/nematrix/snap/reports/dual_arm_fold_towel_2026-10-23/conversion_manifest.json` 与其指向的采集 HDF5 读取；默认日志分别写入 `/home/nematrix/snap/deployment_logs/DualArm_FoldTowel_rtc_full.jsonl` 和 `/home/nematrix/snap/deployment_logs/DualArm_FoldTowel_no_rtc_full.jsonl`。

## goHome 先抬升再回位

本目录的 `goHome.py` 流程为：打开双夹爪 → 两臂分别沿基座/WORLD 的 +Z 抬升 → 在高处移到各自目标正上方并调整姿态 → 固定目标 XY 和姿态下降 → 到位后关闭夹爪。goHome 不调用 `Init()`。

夹爪初始化由独立脚本 `init_grippers.py` 执行：连接并使能左右从臂、保持当前关节位置，两侧夹爪各调用一次 `Init()`，不执行 home 路径、不连接相机、不读取采集数据。RDK 1.9.2 的 `Init()` 返回仅代表请求送达，没有初始化完成标志，因此先等待至少 10 秒，再要求两侧夹指连续 0.5 秒停止运动；总等待上限 30 秒。初始化请求失败、机械臂故障或等待超时均中止，不自动重发 `Init()`。

```bash
python init_grippers.py --check   # 只检查配置，不连接设备
python init_grippers.py           # 单独初始化两侧夹爪，不自动回 home
```

每侧经过高度为 `max(当前 TCP Z, 采集首帧 TCP Z + lift_m)`，默认以最终 home 高度为基准加 0.10 m。当前低于此高度则先抬到此高度；当前已更高则保持当前高度平移，到目标正上方后再下降，不再额外向上抬 10 cm。两臂都完成一段才进入下一段。goHome 的名义 TCP 路径速度从 0.05 提高到 0.075 m/s（7.5 cm/s），实际速度受逐点到位等待及关节速度/加速度限制约束。此值在 `go_home_path.py` 的 `TCP_SPEED_M_S` 中设置。

```bash
python goHome.py --check          # 读取目标和抬升参数，不连接硬件
python goHome.py                  # 以 home 高度 + 10 cm 回位，当前更高则保持高度平移
python goHome.py --lift-m 0.15    # 高度余量改为 15 cm，允许 2–30 cm
```

目标 TCP 与目标关节从采集首帧同一时间戳读取。移动前用 RDK `Model.reachable(..., free_orientation=False)` 预检两臂整条路径；工具标定不匹配、不可达、关节越界或 IK 跳变均中止，不回退原关节插值路径。TCP 路径以不超过 5 mm / 0.03 rad 的点离散为关节目标，每段关节变化不超过原配置的 0.05 rad；下降时逐渐恢复采集首帧的冗余关节姿态。运行中同时检查关节与 TCP 到位，TCP 低于该段高度范围超过 3 mm 时停止。

每个路径点须同时满足 TCP 位置误差不超过 3 mm、姿态误差不超过 0.03 rad、关节误差不超过配置中的 0.02 rad。单点等待上限 5 秒，总回程上限沿用 `start_pose.timeout_s`。等待超过 1 秒后，每秒输出路径点编号、阶段进度、左右臂未满足项（`position` / `orientation` / `joints`）、各项误差以及实际/目标 TCP XYZ；超时前也会输出完整诊断。超时会停止，不跳过该点。

这是小步 IK 生成的 TCP 路径，段间仍由关节控制器插值；没有整臂、工具或双臂碰撞规划，也未标定桌面高度。10 cm 是路径高度余量，不是已验证的桌面安全距离。本次仅改变本目录 `goHome.py` 的回位方式；模型部署、数据回放以及 `examples/flexiv/goHome.py` 仍保留原回位路径。

接口依据：[RDK 1.9.2 Model（只读 IK 可达性检查）](https://github.com/flexivrobotics/flexiv_rdk/blob/v1.9.2/include/flexiv/rdk/model.hpp)、[RobotStates TCP 位姿格式](https://www.flexiv.com/software/rdk/api/structflexiv_1_1rdk_1_1_robot_states.html)。

## 回放一条示教数据

在部署环境中运行，不带参数列出本地 50 条示教，输入编号或 `scene_0001` 这样的名称选择：

```bash
cd /home/nematrix/snap/lerobot/src/lerobot/policies/pi05/deployment
python replay_data.py
```

程序读取转换清单指向的原始 `scene_*/lowdim/lowdim.h5`，每条 scene 是一次完整采集，包含训练过滤前的轨迹，不按训练 episode 的切分截断。回放的是从臂实测关节位置和记录的夹爪指令；原数据未保存关节控制指令。仅连接配置中的左右从臂及夹爪，不连接主臂、相机，不加载 checkpoint。

选定后先检查数据与从臂序列号、时间戳、采样间隙、单位及关节限位，再自动分段对齐**所选数据的第一帧**。到位后按 Enter 才开始回放，输入 `q` 退出；运行中按 Ctrl+C 停止两台从臂。仍保留起点最大偏差 1 rad 和运行目标相对实测值最大偏差 0.25 rad，超限即停止，不跳过这些检查。请先断开采集遥操作或其他控制程序。

```bash
python replay_data.py --list                              # 仅列出数据
python replay_data.py --scene scene_0001 --check           # 仅检查文件，不连接设备
python replay_data.py --scene scene_0001                  # 指定数据，回位后仍等 Enter
python replay_data.py --scene scene_0001 --speed 0.5      # 半速，时间约加倍
```

默认原速 `--speed 1`，支持 `(0, 1]`。关节按原始时间线插值到 15 Hz，夹爪取当前采样时刻之前最后一条开合指令，末帧也会发送并等待到位。SDK 调用迟到时顺延，保留全部重采样帧，不追赶或跳帧；速度/加速度限制可能延长实际时长，跟随偏差过大会中止。`--check` 检查本地数据，机器人关节限位及实际跟随能力需连接后检查。

## 离线测试

```bash
UV_CACHE_DIR=/tmp/flexivstudio-uv \
/home/nematrix/snap/.conda-envs/lerobot-deploy/bin/uv run --no-project \
  --python /home/nematrix/snap/.conda-envs/lerobot-deploy/bin/python \
  python -m unittest discover -s src/lerobot/policies/pi05/deployment -p 'test_*.py' -v
```
