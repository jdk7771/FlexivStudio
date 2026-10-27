# RTC 高频机械臂状态记录

新增入口 `run_rtc_states.sh` 基于原 `run.sh` 复制，调用独立的 `deploy_rtc_states.py`。原启动脚本、部署入口和硬件适配文件均未修改。新入口复用原 RTC 流程，只在本次进程中使用增加了状态采样的硬件子类。

在部署环境、当前目录运行：

```bash
# 交互选择模型和运行方式，默认 200 Hz 状态采样
bash run_rtc_states.sh

# 四卡 RTC 全模型 25k，连续运行，200 Hz
bash run_rtc_states.sh DualArm_FoldTowel_pi05_rtc_full 025000 --run-mode continuous

# 500 Hz，指定保存父目录
bash run_rtc_states.sh DualArm_FoldTowel_pi05_rtc_full 025000 --state-hz 500 --state-dir /home/nematrix/snap/deployment_logs/robot_states

# 只检查文件，不连接设备或创建状态记录
bash run_rtc_states.sh DualArm_FoldTowel_pi05_rtc_full 025000 --check
```

只给步数时默认组为 `DualArm_FoldTowel_pi05_rtc_full`，也可用 `--checkpoint` 指定任意兼容的 RTC 权重。`--state-hz` 范围 1–1000，默认 200；这是主机读取目标频率，不保证控制器每次都更新了状态，1000 Hz 也不意味着能达到稳定 1000 Hz。保持原 15 Hz 策略动作发送、回 home、等待 Enter 和停止逻辑。

**不保存图像**；相机仍供模型推理使用。此入口拒绝 `--record`，并强制关闭原图像记录选项。原 15 Hz telemetry 仍保存动作目标，可用 `wall_time` 与高频状态关联；高频记录没有逐条重复保存动作目标。

状态采样在两侧从臂连接完成后开始，包含回 home、等待按键、策略执行，退出时结束。默认保存到 `telemetry` 同级目录 `robot_states/<UTC时间>_rtc_states_<随机标识>/`：

- `metadata.json`：实际配置、checkpoint、请求采样率、单位和说明。
- `states.jsonl`：左右臂 `q`（rad）、SDK 实测 `dq`（rad/s）、数值估计 `ddq_estimate`（rad/s²）、关节力矩/外力矩、TCP 位姿/速度/外力，以及夹爪宽度/力/运动状态。
- `summary.json`：实际主机采样频率、不同机器人时间戳数量、丢弃行数、错过的采样周期、读取耗时和错误信息。

每侧独立读取一次 `Robot.states()`，保存机器人纳秒时间戳和主机读取起止时间。左右臂及夹爪并非硬件同步采样；机器人时钟与主机时钟可能不同。重复 SDK 时间戳标记为 `duplicate_timestamp=true`。加速度按相邻不同机器人时间戳的 `dq` 差分计算，重复、倒退或超过 100 ms 的时间间隔记为 `null`；这是含噪声的估计，不是传感器实测加速度，分析时应平滑。

专用采样线程只读取已有 RDK 连接的状态；写盘在另一个线程完成，队列最多 2048 行，满时丢新行而不阻塞控制。读取或写盘失败会停止状态记录并输出错误，不自动停止策略。退出时先通知采样停止、停止机械臂，再限时排空记录队列。没有额外机器人连接或运动命令。

线程仍会占用 CPU/GIL 和磁盘带宽，不能承诺完全零开销。先用 200 Hz，依据 `summary.json` 的实际频率、重复时间戳和原控制周期日志判断是否适合提高频率。强制杀进程或断电可能缺少尾部记录或汇总。
