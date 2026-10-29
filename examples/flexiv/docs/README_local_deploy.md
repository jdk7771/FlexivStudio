# DualArm_FoldTowel 本地部署

入口：`examples/flexiv/deploy_dualarm.py`；配置：`examples/flexiv/deploy_config.json`。
复用本仓库 `RTCInferenceEngine`，推理使用 **`mode="trained"`**，匹配三个训练任务的 training RTC，支持 LoRA 和冻结视觉的全量微调 checkpoint。没有改动训练代码或运行中的任务。

## 与真机、数据对应

依据原始 `template_test/metadata.yaml`、采集包 `configs/dual_arm_test.py`，以及采集镜像中的 SDK 方法签名、夹爪/相机源码编写。采集镜像使用 Flexiv RDK 1.9.2 和 TDK 1.6.2；部署只使用 RDK 控制从臂，不接入 TDK 遥操作。

| 设备     | 序列号           | 部署用途                    |
| -------- | ---------------- | --------------------------- |
| 左从臂   | `Rizon4-063906`  | 左侧 7 关节和夹爪           |
| 右从臂   | `Rizon4R-062106` | 右侧 7 关节和夹爪           |
| 全局相机 | `350122061994`   | `observation.images.image`  |
| 左腕相机 | `840412062188`   | `observation.images.image2` |
| 右腕相机 | `350122063150`   | `observation.images.image3` |

两台主臂 `Rizon4-063848`、`Rizon4R-062103` 不建立连接。网卡白名单默认沿用采集配置的 `192.168.1.32`，实际换网卡时修改 JSON。工具 `Teleop-GN01`、夹爪 `Flexiv-GN01` 也沿用采集配置。

连接时会打印机器人当前工具和可用工具列表；当前已选中配置工具时直接沿用，使能后不再重复查询或切换。只读工具查询遇到 RDK `No reply from the robot` 时，在 `hardware.connect_timeout_s`（默认 15 秒）的窗口内间隔重试并打印从臂序列号及查询名称；持续无回复则中止，其他查询异常、断连或故障直接报错。工具切换、使能和运动指令不使用此重试。配置工具缺失会在使能前中止并列出工具名，需核对 Flexiv Elements 中实际安装工具的参数和名称。

两台从臂都须在 Flexiv Elements 中切到 Auto 自动模式，急停释放且无故障后才能使能。程序会输出每台从臂的序列号和 operational status；检测到 Manual 手动模式会立即中止，不能通过延长超时解决，也不会自动切换自动模式。连接按左、右顺序进行：若只出现 `Rizon4-063906` 的日志，说明尚未连接右从臂。

`goHome.py` 在起点检查通过后，先给左右夹爪各明确发送一次打开指令，打开到配置最大开度（当前 0.1 m）并等待到位；保持打开完成关节回位，最后给两侧各明确发送一次闭合指令。显式开合不受普通推理动作的 1 mm 指令死区限制。结束时校验的是 RDK 报告的宽度与运动状态；若读数为零而目视仍有开口，仍需检查夹爪零位、安装和反馈，不能仅凭读数判断物理闭合。

状态/动作固定顺序：`left_joint_1..7, left_gripper, right_joint_1..7, right_gripper`。关节单位 rad、夹爪单位 m。
图像以 1280×720 / 15 Hz 采 RGB，与转换脚本相同地用 PIL 双线性缩放、居中补黑边至 224×224；三台相机各有读取线程，控制循环取最新帧，检查接收时间的过期和偏差。这是软件时间检查，没有宣称相机硬件同步。

RTC 引擎在设备连接前完成构造，避免预处理器/tokenizer 复制期间阻塞已启动的相机线程。实时相机帧龄上限仍为 200 ms；报错会列出三路相机名称、序列号及各自帧龄，连续运行的 JSONL 也记录 `camera_ages_s`。超限仍停止机械臂，不以放宽阈值掩盖停帧。

设备连接并完成首帧对齐后，先自动等待 2 秒稳定。连续模式保持在 home（采集首帧姿态），按 Enter 后才启动 RTC 线程，输入 `q` 退出。随后程序自动等待三路相机各收到一张按键之后产生的新帧，并通过原有帧龄/时间偏差检查，再允许策略运行。单步模式在初始化就绪后等待每步的 Enter，每次按键后也重新获取新帧。新帧等待上限沿用 `hardware.connect_timeout_s`（15 秒），等待期间不下发策略动作；超时中止并保留相机详情，读取线程异常或机器人故障直接报错。

流程：三路图像 + 实测关节/夹爪 → checkpoint 预处理和归一化 → BF16 PI05 → 反归一化、以本次观测为起点还原绝对关节 → 15 Hz 下发从臂与夹爪。
`delta[k] = q_target[k] - q_observation`，整个 chunk 共用一个起点；禁止再次累计 joint delta。RTC 引擎负责把上一块剩余绝对动作转换到新观测起点，再作为前缀传入模型。夹爪始终为绝对开度。

默认 chunk 50、训练最大前缀 8、推理前缀容量 10、队列剩余 25 步开始补充、去噪 10 步。8 步对应约 0.53 秒；持续超过训练前缀范围的延迟会被引擎拒绝，不能靠继续增加推理前缀解决。
`max_observation_age_s=5` 是引擎对动作来源观测的有效期，包含整个动作块播放；相机实时过期检查单独是 0.2 秒。

关节调用 RDK 1.9.2 的 `SendJointPosition(q, dq, max_vel, max_acc)`，默认使用 `NRT_JOINT_POSITION`，由机器人内部轨迹生成器平滑，不在 Python 中模拟 1 kHz 伺服。RDK 会在重复下发目标时重新规划，因此暂停时只下发一次当前位姿保持指令。默认速度 0.5 rad/s、加速度 1 rad/s²、目标离实测关节超过 0.25 rad 则报错；这些是初次部署限制，需按实际动作调整，可能限制复现速度。夹爪沿用采集的 0.2 m/s、开 -65 N / 合 +45 N，目标变化不足 1 mm 不重复下发。
不调用机器人 home、夹爪 Init，也不自动清机器人故障。每次真机运行前，程序从原始采集 HDF5 和转换清单读取 `scene_0001` 第一个**实际进入训练集**的观测帧（时间戳 `1792553391585.0` ms），检查当前关节与该帧的差值后自动向该姿态移动。单关节差值超过 1.0 rad 时拒绝运动，须人工先摆近；对齐时每段目标最多变化 0.05 rad，逐段等到位，最后对齐夹爪开度。源 HDF5/转换清单缺失、超时或姿态越界均中止运行。此过程只按关节空间移动，没有碰撞检测，操作前必须清空工作空间并确认毛巾/夹爪状态。退出或报错时停止两台从臂，保持夹爪最后指令。

## 可配置的关节接触力矩上限

修改同目录 `deploy_config.json`：将 `hardware.control_mode` 改为 `"NRT_JOINT_IMPEDANCE"`，再给 `hardware.arms.left.max_contact_torque_nm` 和 `hardware.arms.right.max_contact_torque_nm` 分别填写 7 个数，顺序为 A1–A7，单位 Nm，重启程序后应用。每个值必须在 `[0, RobotInfo.tau_max 对应关节值]` 内；`tau_max` 是机器人上报的关节力矩软件上限，不代表当前接触力矩设置。`null` 表示该侧不调用设置接口，程序不会因为填写了数值而自动更改控制模式。

连接时在切换到关节阻抗模式后、该臂第一次 `SendJointPosition` 之前调用 `Robot.SetMaxContactTorque`，并打印该侧生效请求值。配置为位置模式时填写力矩会直接报错；超出机器人额定范围或 SDK 拒绝设置时中止。部署和 `goHome.py` 共用此配置。阻抗模式会改变接触响应，需要重新验证任务表现。

这是关节阻抗控制器的接触力矩调节参数，**不会修改 311016 安全板 TCP force 停机阈值**。接口依据：[RDK 1.9.2 robot.hpp](https://github.com/flexivrobotics/flexiv_rdk/blob/v1.9.2/include/flexiv/rdk/robot.hpp#L533)。

## 本地 Conda 环境

本机已在 `/home/nematrix/snap/.conda-envs/lerobot-deploy` 建好 **Python 3.12** Conda 环境；采集容器自身是 Python 3.10，不能直接当 LeRobot 环境使用。
本机 RTX 5090 / 32 GB、驱动 580.178.04。已安装 LeRobot 0.6.2、PyTorch 2.7.1+cu128、Transformers 5.5.4、PEFT 0.21.2、Flexiv RDK 1.9.2 和 pyrealsense2 2.59.1.11053。已验证 CUDA BF16 运算和 PI05/trained RTC/LoRA/SDK 导入；部署加载后还会检查策略参数全部位于 CUDA。

直接激活现有环境：

```bash
conda activate /home/nematrix/snap/.conda-envs/lerobot-deploy
```

新机器可运行 `bash examples/flexiv/setup/setup_local_deploy.sh` 重建环境；该脚本会拒绝覆盖现有路径。环境及安装缓存都放在 `/home/nematrix/snap` 下。脚本安装 CUDA 12.8 的 PyTorch 2.7.1，与服务器 PyTorch 主版本一致；**不要沿用服务器的 cu118 安装命令**。机器人软件 3.11.1 对应 RDK 1.9.2，采集容器也使用 1.9.2；不混用本机 base 环境的 RDK 2.1。
安装 `lerobot[pi,peft]`、RealSense、RDK 和 `h5py` 即可。`h5py` 用于每次启动读取原始采集首帧；实时 RGB 输入不经过视频文件，不需要 torchcodec/视频解码依赖。

## 准备 checkpoint

三个训练任选一个**已保存完成的具体 step**，将 `checkpoints/<step>/pretrained_model/` 整个目录复制到本地配置指定位置。不要只复制 `adapter_model.safetensors`。
默认目标是 `/home/nematrix/snap/models/DualArm_FoldTowel/pretrained_model`，目前该链接指向 E1 的 12,500-step 版本。三个实验的最新与最佳已验证保存权重均已下载，具体 step、路径和 SHA256 见 [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md#最新与最佳已保存权重2026-10-23)。无参数运行且不传 `--checkpoint` 时，会先列出本地各实验/step，输入编号后才加载权重和连接设备；输入 `q` 可退出。传 `--checkpoint /实际/pretrained_model` 则跳过菜单。E1/E3 LoRA 使用 `/home/nematrix/snap/models/pi05_base/model.safetensors`（14.47 GB），其 SHA256 为 `0eb11ca9587678c1d2ef8cf32807c29f8ce53a2bfdfc1aa4a4c96f16fca59b0f`，与 [LeRobot 官方 pi05_base](https://huggingface.co/lerobot/pi05_base/blob/main/model.safetensors) 一致。缺少配置的 base 会直接报错，不再回退到 `/tmp`；E2 全量权重无需 base。

| 训练                            | 服务器输出目录（1020 端口）                                        |
| ------------------------------- | ------------------------------------------------------------------ |
| GPU 4 LoRA / batch 16           | `/mnt/work/dakun/logs/projects/lerobot/DualArm_FoldTowel_pi05_rtc_lora_run7`     |
| GPU 5、6 全量 / global batch 16 | `/mnt/work/dakun/logs/projects/lerobot/DualArm_FoldTowel_pi05_rtc_full_gpu5_6`   |
| GPU 7 LoRA / batch 8            | `/mnt/work/dakun/logs/projects/lerobot/DualArm_FoldTowel_pi05_rtc_lora_gpu7_bs8` |

必须保留 `config.json`、`policy_preprocessor.json`、`policy_postprocessor.json` 及其引用的所有统计 `.safetensors`。
LoRA 还要 `adapter_config.json`、`adapter_model.safetensors` 和**同一个原始 PI05 base**。全量模型使用其自己的 `model.safetensors`，无需额外 base。
两者均需本地 PaliGemma tokenizer 完整目录；优先复制 checkpoint 自带的 tokenizer 文件，或服务器 `/mnt/work/dakun/model/paligemma-3b-pt-224`。
base 已放在 `/home/nematrix/snap/models/pi05_base`；换机器时须重新提供这份权重或修改配置。tokenizer 默认直接使用 checkpoint 内的 `tokenizer/`，换 checkpoint 也会自动选择其内置目录；需要时可用 `--tokenizer` 指定独立目录。服务器的绝对路径不会被直接使用。
加载使用严格权重校验；文件缺失、维度/顺序不符或统计缺失立即失败。读取模型时强制离线，避免悄悄联网下载。

## 先离线检查，后真机

只检查本地配置和文件，不需要完整推理环境，也不加载模型/SDK：

```bash
python examples/flexiv/deploy_dualarm.py --check
```

可用 `--checkpoint /实际/pretrained_model --base-model /实际/pi05_base --tokenizer /实际/tokenizer` 覆盖路径。

可选的单次离线预测（需要完整环境和 GPU，但不接设备）：

```bash
python examples/flexiv/deploy_dualarm.py \
  --predict-npz /path/to/observation.npz --output /path/to/new_absolute_actions.npy
```

NPZ 必须含 `observation.state`：`float32[16]`，以及三个 `observation.images.image/image2/image3`：已按上述方式处理的 `uint8[224,224,3]` RGB。
输出为绝对物理单位的 `[50,16]`。这是无前缀的首块预测检查，不代表已经验证真机闭环或 RTC 接续。

让双臂回到采集数据的第一帧关节姿态，同时闭合双夹爪（不加载模型、不打开相机、不执行策略）：

```bash
cd /home/nematrix/snap/lerobot
python examples/flexiv/goHome.py --check  # 只打印首帧目标，不连接设备
python examples/flexiv/goHome.py          # 检查当前位置并直接分段移动
```

`goHome.py` 与部署共用 `deploy_config.json` 的 1.0 rad 起点距离上限、每段 0.05 rad 目标增量及 60 秒回位超时；超限时列出关节当前值和首帧目标值并拒绝运动。通过检查后，先打开双夹爪到 0.1 m 并等待最多 10 秒到位；打开失败则不开始关节回位。随后保持夹爪打开，分段回到采集首帧关节位置，到位后再关闭双夹爪到 0 m。RDK 的 `Move()` 只保证指令送达，因此关闭后等待最多 10 秒，确认左右实测开度均不大于 1 mm 且不再运动；否则报出两侧开度、运动状态和力值。无参数运行直接执行，不再等待确认；采集程序若仍在发送机械臂指令，须先停止采集遥操作。退出时停止两台从臂，不关闭或占用相机。

**以下命令选完 checkpoint 和运行方式后会自动移动到采集首帧，到位后等待 Enter 才执行策略**：

```bash
cd /home/nematrix/snap/lerobot
python examples/flexiv/deploy_dualarm.py
```

运行后先选择本地 checkpoint，再选 `1` 单步或 `2` 连续。连接设备并通过起点距离检查后会自动移动到采集首帧。单步模式每按一次 Enter 才从最新相机/关节观测生成并下发**一个**动作，等到位后暂停并清空 RTC 队列，输入 `q` 退出；每步重新推理，因此是谨慎检查模式，不等同连续 RTC 的时间行为。连续模式对齐后保持姿态并等待 Enter，按键后通过新帧就绪检查，才按 15 Hz 执行策略；默认运行 60 秒，从就绪检查完成时开始计时，可改 JSON 的 `duration_s`。等待启动时输入 `q` 可退出，两种模式都可用 Ctrl+C 停止。运行前在采集网页断开设备，避免相机占用或采集遥操作继续发指令；并先让机械臂处于接近首帧的安全位置。

也可用 `--run-mode step` 或 `--run-mode continuous` 跳过方式菜单；用 `--checkpoint /实际/pretrained_model` 跳过 checkpoint 菜单。两者都指定时仍会先自动回位，再等待 Enter 才执行策略。
每步状态、绝对目标、相机年龄、读取/下发时间和队列长度写入配置的 JSONL；不要在日志里把推理输出再解释成逐步增量。

## 本次验证范围

执行了 Conda 环境的依赖检查、5090 CUDA BF16 运算、PI05/trained RTC/LoRA/RDK/RealSense 导入、脚本语法检查，以及纯函数和虚拟 RDK 接口测试：维度与单位、状态/绝对目标映射、图像处理、坏 checkpoint 拒绝、目标越界拦截和部分连接失败清理。新交互还以模拟机械臂验证了首帧分段移动、超距拒绝和单步只下发一次动作；原始 HDF5 读出的首帧与数据转换器输出逐元素一致。真机运动与 RTC 速度仍未实测。
E1/E2/E3 已完成离线严格权重加载，参数全部位于 `cuda:0`；E1 用合成观测完成一次离线前向，并再次从持久化 base 路径严格加载。三路 RealSense 已在本机同时打开、各读取一帧并关闭；RDK 1.9.2 已只读连接两台从臂，均上报软件 `v3.11.1`、7 自由度和 7 个关节限位。没有使能、切控制模式或下发动作。闭环效果和实际速度仍需之后上机验证。
