# OpenPI π0.5 双臂 TCP pose delta + RTC 全量微调

独立入口：`training_pose_delta/train.py`。模型使用现有官方 OpenPI `PI0Pytorch` 和
RTC 补丁；视觉编码器、语言模型、动作专家全部更新，无 LoRA。复用已有 `.venv`。
本目录的训练器与数据适配器独立于 `training/` 下正在运行的 joint 实验。

## 已准备的文件

| 文件 | 用途 |
| --- | --- |
| `configs/dualarm_fold_towel_pose_delta_rtc_full.json` | 实验参数、pose 定义、数据/输出路径 |
| `train_dualarm_fold_towel_pose_delta_rtc_full.sh` | 数据准备、CPU 检查、测试、四卡训练、续训 |
| `train.py` | DDP 全量训练、14 维 RTC loss、验证与完整状态保存 |
| `runtime.py` | 16 维绝对 TCP → 20 维状态 / 14 维 delta、归一化和 tokenizer |
| `pose_math.py` | 旋转组合、SLERP、pose delta 和反变换 |
| `prepare_dataset.py` | 利用已有视频对齐，从原始 HDF5 生成独立 pose 数据集 |
| `test_pose_delta.py` | CPU 几何、转换、归一化、模型梯度测试 |

## 数据与动作定义

完整原始数据位于
`/mnt/work/dakun/DualArm_FoldTowel_joint_rtc/data/template_test`。
顶层 `/mnt/work/template_test` 的 lowdim 文件不完整，不使用该副本。
左臂为 `063906`，右臂为 `062106`。`tcp_pose_*` 按 `[x,y,z,qw,qx,qy,qz]` 读取，
采用采集工具的 wxyz 约定（本机 `data_manager_v0.4.6/scripts/view_scene_open3d.py`
同样以 `pose7[3:7]` 作为 qw,qx,qy,qz）；xyz 单位米。

新数据集：`/mnt/work/dakun/DualArm_FoldTowel_pose_rtc`。
沿用 joint 数据集的 123 个 episode、28,622 帧、15 Hz 和三路 224×224 视频，
112 个 episode 训练、11 个验证。视频通过只读使用的符号链接复用；新 parquet、
metadata 和统计文件写入独立目录。转换不会写入源数据集或视频。

存储的 state/action 均为每臂 8 维、共 16 维绝对值：

```text
left:  x y z qw qx qy qz gripper
right: x y z qw qx qy qz gripper
```

state 使用 t 时刻测量的 TCP；action 使用 t+1/15 秒的测量 TCP，夹爪使用 t 时刻的
绝对命令。位置做线性插值，姿态做最短路径 SLERP。原始数据没有记录 TCP 目标命令，
所以这是 **下一时刻实测位姿作为监督目标**。夹爪沿用已有对齐数据中的精确标签；
重建时间网格后先与原始 HDF5 交叉检查，容许测量状态最多 5 微米的浮点重建误差。

模型看到的动作共 **14 维**：每臂 `[dx,dy,dz,drx,dry,drz,gripper]`。
默认 `pose_delta.delta_frame="base"`：

```text
Δp[t,k] = p_target[t+k] - p_state[t]
Δr[t,k] = Log(R_target[t+k] @ R_state[t].T)  # 旋转向量，弧度
gripper[t,k] = absolute_gripper_command[t+k]
```

整个 50 步 chunk 共享当前观测 t 的原点。位置和旋转差在各自机械臂的基坐标系表示；
两臂没有假设共享 world 外参。姿态通过旋转组合求差，避免直接相减欧拉角或四元数。
episode 末尾和 LeRobot reader 一样重复最后一个 action，不跨 episode。

`delta_frame="tcp"` 也受支持：`Δp=R_state.T@(p_target-p_state)`，
`ΔR=R_state.T@R_target`。切换时必须选择新的数据目录并重新准备统计，不能直接续训。

模型 state 共 **20 维**，每臂 `[xyz, R第一列, R第二列, gripper]`，
避免绝对四元数正负号跳变。state 及 action 都 pad 到模型的 32 维。
训练/验证 loss 只计算实际 14 维动作；RTC delay 为 0…8，干净前缀不计入 loss。

归一化文件 `meta/pose_delta_stats.json` 只由训练 episodes 拟合：
state 先转成 20 维；action 先按所有训练 anchor 的 50 步 chunk 计算 delta，再求 q01/q99。
验证集不参与拟合。常量维度归一化为 0，反归一化回训练常量。
这份统计与普通 `meta/stats.json`（存储的绝对 16 维数据统计）用途不同。
相较现有 joint 实验，动作单位、state 表示和归一化统计均不同，loss 数值不直接可比。

## 检查和启动

```bash
cd /mnt/work/dakun/lerobot/src/lerobot/policies/openpi

# 只显示命令
bash training_pose_delta/train_dualarm_fold_towel_pose_delta_rtc_full.sh --print-command

# CPU：真实 pose 数据、视频、tokenizer、预训练权重形状
bash training_pose_delta/train_dualarm_fold_towel_pose_delta_rtc_full.sh --check-data

# CPU：几何/转换/归一化以及缩小后的官方模型前向反向
bash training_pose_delta/train_dualarm_fold_towel_pose_delta_rtc_full.sh --test
```

当前机器的 GPU 0–3 正在进行 joint 续训。**新 pose 训练尚未启动**。
待所选四张卡空闲后，再手动执行：

```bash
cd /mnt/work/dakun/lerobot/src/lerobot/policies/openpi
CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_PORT=29525 \
nohup bash training_pose_delta/train_dualarm_fold_towel_pose_delta_rtc_full.sh \
  >> /mnt/work/dakun/outputs/train_openpi_pose_delta_rtc_full.log 2>&1 &
echo "PID: $!"
tail -f /mnt/work/dakun/outputs/train_openpi_pose_delta_rtc_full.log
```

默认每卡 batch 8、4 卡全局 batch 32、全量微调 50,000 步、每 2,500 步保存、
每 1,000 步验证。AdamW 学习率 1e-5，warmup 1000，使用 bf16 和梯度检查点。
初始模型为 `/mnt/work/dakun/models/pi05_base`，pose 实验从头建立优化器和调度进度。
输出目录：
`/mnt/work/dakun/outputs/openpi_pi05_dualarm_fold_towel_pose_delta_rtc_full`。
日志放在输出目录外，避免首次训练的“非空输出目录”检查。

配置已经生成的数据集可直接使用。只有需要另建数据集时才执行准备命令
（已有目录会报错，避免覆盖；修改 JSON 路径后可通过 `OPENPI_CONFIG` 指定）：

```bash
nice -n 19 bash training_pose_delta/train_dualarm_fold_towel_pose_delta_rtc_full.sh --prepare-data
```

准备过程隐藏 GPU，并限制数值计算到一个 CPU 线程。全量训练资源消耗需在正式训练时评估；
CPU 缩小模型检查不代表已完成四卡全尺寸训练验证。

## 续训和输出还原

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash training_pose_delta/train_dualarm_fold_towel_pose_delta_rtc_full.sh --resume latest
```

也可使用 `--resume /绝对路径/checkpoints/002500`。只接受 pose 实验检查点，
恢复模型、AdamW、step/epoch/batch offset、各 rank RNG，并检查配置和 pose 统计未变化。
`pose_delta_stats.json` 同时保存到运行根目录与各检查点，便于推理重现训练变换。
恢复后 CSV 追加写入，若停机前有未保存步数，重复 step 的指标应取最后一条。

`Collator.denormalize_actions()` 将网络输出还原为物理单位的 14 维 delta；
`pose_math.apply_pose_delta(current_pose, deltas, frame)` 再还原为绝对 TCP 四元数位姿。
批量 chunk 使用 `current_pose[:, None, :]`。这里提供数学还原，不包含机器人控制入口。
实机 RTC 推理仍需把旧动作前缀转换到当前观测原点，并接入硬前缀钳制；
现有 joint 部署脚本不能直接消费 pose 模型。
