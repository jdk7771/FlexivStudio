# DualArm_FoldTowel 训练与测试记录

本记录区分训练日志、离线验证和真机效果。指标快照时间为 **2026-10-23 11:50（北京时间）**；训练仍在继续，表中的 step 和 loss 不是最终结果。服务器为 `101.132.150.51:1020`，用户 `dakun`。不在本文件记录登录凭据。

## 数据与共同配置

- 原始数据：`/media/nematrix/50AA6D45AA6D28A43/data/template_test`；服务器训练集：`/mnt/work/dakun/DualArm_FoldTowel_joint_rtc`，任务文本 `DualArm_FoldTowel`。转换清单状态为 `complete`：50 个 scene、123 个 episode、28,622 帧，15 Hz，三路图像 224×224。
- 训练/验证按 **完整 scene** 划分：最后 5 个 scene（11 个 episode）只用于验证；其余 45 个 scene（112 个 episode）用于训练。`eval_split=0.08536585365853659`。同一 scene 的切段不会跨训练与验证集；这不是独立采集的新场景测试。
- 状态与动作均为 16 维，顺序是左臂 7 关节、左夹爪、右臂 7 关节、右夹爪。关节单位 rad，夹爪单位 m。原始动作存储为下一时刻实测关节绝对位置及夹爪绝对指令；训练预处理把关节变为相对当前观测的 delta，夹爪保持绝对值。同一个 50 步 chunk 共用观测起点。
- 三组均从同一 PI05 base 开始：BF16、chunk 50、training RTC 最大前缀延迟 8、梯度检查点、50,000 次优化器更新。训练 loss 每 50 step 记一次，验证 loss 每 1,000 step 计算一次，每 2,500 step 保存 checkpoint。视频后端为 `torchcodec`；日志里 `data_s` 通常约 0.001 s，训练耗时主要在模型更新。
- 训练环境：LeRobot 0.6.2、PyTorch 2.7.1+cu118、Transformers 5.5.4、PEFT 0.21.2。服务器代码目录没有 Git 元数据；下列三个关键文件与本地工作区 SHA256 一致：`lerobot_train.py` 为 `61de8a5794a149bd9e0cfe409178d58e5cebfc3d288fa0c04b27cdf8b7e83c02`，`modeling_pi05.py` 为 `7cf56d77f29a0d3564bde7e7e1fa7333fffd0d81472d2d393cf2afea6967716c`，`datasets/factory.py` 为 `ee9580a1d17e6654f3ce18eec852022f229cb05a754df2fbc73431128c1c1f2b`。

数据证据：[转换清单](../../../../reports/dual_arm_fold_towel_2026-10-23/conversion_manifest.json)、[loss 曲线](../../../../reports/dual_arm_fold_towel_2026-10-23/loss_overview.png)、[解析后的指标](../../../../reports/dual_arm_fold_towel_2026-10-23/summary.json)。曲线中的浅色线是每 50 step 的训练 loss 平均值，深色线是 5 点平滑，红点线是验证 loss，灰色竖虚线是当时最新 checkpoint。验证 loss 不是折毛巾成功率。

本地实时网页：[loss_dashboard.py](../../../../reports/dual_arm_fold_towel_2026-10-23/loss_dashboard.py) 由 [用户服务](../../../../reports/dual_arm_fold_towel_2026-10-23/dakun-loss-dashboard.service) 保持运行，访问 `http://127.0.0.1:8765`。它每 60 秒通过 SSH 只读同步三份服务器训练日志和最新 checkpoint，网页显示训练/验证 loss、step，并提供总览、单实验曲线和 CSV 下载。SSH 断开时网页仍显示本地快照及同步错误。若重启电脑或连接断开，先在本机终端执行 `ssh -p 1020 -M -S /tmp/codex-master -N dakun@101.132.150.51` 并交互输入登录密码；保持该终端打开即可恢复实时同步。密码不会写进脚本。

```bash
systemctl --user status dakun-loss-dashboard.service
systemctl --user restart dakun-loss-dashboard.service
```

## 三组实验

| ID  | 方法与 GPU        | Batch / 学习率           | 可训练范围                                                                                              | 服务器输出目录                                                     |
| --- | ----------------- | ------------------------ | ------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| E1  | LoRA，GPU 4       | 全局 16 / `1e-4`         | 冻结视觉编码器和 PaliGemma 主干；动作专家的 `q_proj`、`v_proj` 及动作/时间投影加 LoRA，`r=32, alpha=32` | `/mnt/work/dakun/outputs/DualArm_FoldTowel_pi05_rtc_lora_run7`     |
| E2  | 全量微调，GPU 5+6 | 每卡 8、全局 16 / `1e-5` | 冻结视觉编码器；其余 VLM 与动作专家参数参与全量微调                                                     | `/mnt/work/dakun/outputs/DualArm_FoldTowel_pi05_rtc_full_gpu5_6`   |
| E3  | LoRA，GPU 7       | 全局 8 / `1e-4`          | 与 E1 相同的 LoRA 范围、rank 和 alpha                                                                   | `/mnt/work/dakun/outputs/DualArm_FoldTowel_pi05_rtc_lora_gpu7_bs8` |

配置证据：[E1 保存配置](../../../../models/DualArm_FoldTowel/experiment1/012500/pretrained_model/train_config.json)、[E2 保存配置](../../../../reports/dual_arm_fold_towel_2026-10-23/train_config_full_gpu5_6.json)、[E3 保存配置](../../../../reports/dual_arm_fold_towel_2026-10-23/train_config_lora_gpu7.json)。E1 adapter 有 80 个 LoRA 张量、2,705,408 个参数，约 10.8 MB；它必须和原始 PI05 base 一起用于推理。

## 训练快照

2026-10-23 11:50 北京时间，日志仍在更新：

| ID  | 最新训练 step / loss | 最新验证 step / loss | 已观察到的最低验证 loss | 最新已保存 checkpoint | 约每步耗时 |
| --- | -------------------- | -------------------- | ----------------------- | --------------------- | ---------- |
| E1  | 14,000 / 0.041       | 14,000 / 0.0539      | 0.0529 @ 12,000         | 12,500                | 2.79 s     |
| E2  | 5,050 / 0.037        | 5,000 / 0.0568       | 0.0565 @ 2,000          | 5,000                 | 1.91 s     |
| E3  | 3,800 / 0.061        | 3,000 / 0.0691       | 0.0691 @ 3,000          | 2,500                 | 1.46 s     |

E1 的验证 loss 在 12,000 step 附近最低，14,000 step 为 0.0539；E2 的训练 loss 持续下降，但验证 loss 在 2,000 至 5,000 step 没有同步下降；E3 的验证 loss 从 1,000 step 的 0.0929 降到 3,000 step 的 0.0691。三组所处训练阶段不同，不凭单个 loss 点判断真机优劣。原始日志、CSV 和重画脚本保存在 `/home/nematrix/snap/reports/dual_arm_fold_towel_2026-10-23/`。

## 最新权重下载

2026-10-23 13:46 北京时间，核对服务器的 `checkpoints/last` 后下载了以下三个 `pretrained_model` 目录。均经过 `rsync --checksum --dry-run` 与服务器逐文件比对，且本地 `deploy_dualarm.py --check` 通过；此检查没有加载模型或连接真机。

| ID  | checkpoint | 本地 `pretrained_model`                                                            | 主权重 SHA256                                                      |
| --- | ---------- | ---------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| E1  | 15,000     | `/home/nematrix/snap/models/DualArm_FoldTowel/experiment1/015000/pretrained_model` | `6434237993aa70dca16453166b066ec5ef8b93c1cbed65b41b3194ec51c3eb26` |
| E2  | 7,500      | `/home/nematrix/snap/models/DualArm_FoldTowel/experiment2/007500/pretrained_model` | `bf193810a3d0352d5672f9cb03a7ca19ff8edebaf777ff45fb3eb2ff0b84831f` |
| E3  | 7,500      | `/home/nematrix/snap/models/DualArm_FoldTowel/experiment3/007500/pretrained_model` | `941eee13dd0adcc8de72d94cd191846b48cd8fc84ef9ff60ab4dcacfb4b41e47` |

E1/E3 主权重为 `adapter_model.safetensors`，各目录含 tokenizer、预处理配置及统计；E2 主权重为约 9.35 GB 的 `model.safetensors`。未下载 `training_state`/优化器状态，故这些目录用于部署测试，不是本地恢复训练的完整 checkpoint。原有 `/home/nematrix/snap/models/DualArm_FoldTowel/pretrained_model` 链接仍指向 E1 的 12,500-step 版本，未自动切换测试模型。LoRA 推理仍需匹配的 PI05 base。

### 最新与最佳已保存权重（2026-10-23）

再次核对 1020 训练服务器后，每组下载最新已完成的 checkpoint，以及**验证 loss 恰好记录在保存 step**时的最低值对应 checkpoint。验证 loss 更低但未保存的 E1 19,000（0.0498）、E2 7,000（0.0548）和 E3 9,000（0.0554）不能直接作为模型下载；保存间隔内的其他 checkpoint 没有同 step 验证值，故“最佳”只是这个明确标准下的候选，不代表真机效果。下载使用临时目录，全部 `pretrained_model` 文件复制完成后才移到正式位置；主权重的本地与服务器 SHA256 完全一致，`deploy_dualarm.py --check` 均通过。未改动服务器训练进程、`checkpoints/last` 或本地默认模型链接。

| 实验    | 选择           | step / 同 step 验证 loss | 本地 `pretrained_model`                                                            | 主权重 SHA256                                                      |
| ------- | -------------- | ------------------------ | ---------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| E1 LoRA | 最佳已验证保存 | 10,000 / 0.0544          | `/home/nematrix/snap/models/DualArm_FoldTowel/experiment1/010000/pretrained_model` | `00f9a9573b1ecedbfe24841ea063d282610a4df58ea6b63806efa39b2bdbd853` |
| E1 LoRA | 最新           | 17,500 / 无同 step 验证  | `/home/nematrix/snap/models/DualArm_FoldTowel/experiment1/017500/pretrained_model` | `970fdf2cf65d04a8d635728593aaee346fb9ec207f1a1af99bf517291d648e89` |
| E2 全量 | 最佳已验证保存 | 5,000 / 0.0568           | `/home/nematrix/snap/models/DualArm_FoldTowel/experiment2/005000/pretrained_model` | `ceef7383f54cbf2591f7b8f2ddd5ef800d05b5dace9b178fa9f2d562cabb1170` |
| E2 全量 | 最新           | 12,500 / 无同 step 验证  | `/home/nematrix/snap/models/DualArm_FoldTowel/experiment2/012500/pretrained_model` | `061a5665480f5729002f52190a9c0434eb9771179f24117f630598a409cfd6b1` |
| E3 LoRA | 最佳已验证保存 | 10,000 / 0.0563          | `/home/nematrix/snap/models/DualArm_FoldTowel/experiment3/010000/pretrained_model` | `8a13c259bf8a1293d158bbc57fcc7185d0165894e0054b0242fde29787b373c9` |
| E3 LoRA | 最新           | 12,500 / 无同 step 验证  | `/home/nematrix/snap/models/DualArm_FoldTowel/experiment3/012500/pretrained_model` | `5718f394d5441d3f5ac4dd3edef4a0ebf9112e2d68a6af1d4`                |

### 2026-10-23 17:43 最新权重下载

本次按请求开始时 1020 服务器三组 `checkpoints/last` 的解析结果，固定下载以下具体 step 的 `pretrained_model`。三组均先下载到隐藏临时目录，通过文件完整性检查后移到正式目录；本地与服务器主权重 SHA256 一致，配置、归一化统计及 tokenizer 经 rsync checksum 比对一致，`deploy_dualarm.py --check` 通过。部署菜单已能自动发现这三个版本。

| 实验                      | 本次下载 step | 本地目录                                                                           | 主权重 SHA256                                                      |
| ------------------------- | ------------- | ---------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| E1 LoRA / batch 16        | 20,000        | `/home/nematrix/snap/models/DualArm_FoldTowel/experiment1/020000/pretrained_model` | `e9543b910885835ba32365016ebe8dbaab8db88c1d0688a2a0cad993f79479e3` |
| E2 全量 / global batch 16 | 15,000        | `/home/nematrix/snap/models/DualArm_FoldTowel/experiment2/015000/pretrained_model` | `a7341aa80c827444bdf46f87887096836f2d934308706e405d6326495ab570ef` |
| E3 LoRA / batch 8         | 17,500        | `/home/nematrix/snap/models/DualArm_FoldTowel/experiment3/017500/pretrained_model` | `236b9e3eaf1cce6f927a94d901e63eb29849fce0dabe38d56b35bf570c5653ab` |

服务器来源分别为上表 E1/E2/E3 对应实验输出目录下的 `checkpoints/020000`、`checkpoints/015000`、`checkpoints/017500`。仅下载部署模型文件，未下载优化器/训练状态；未切换本地默认模型链接，未修改或停止服务器训练，未加载模型或连接真机。

## 已完成检查

| 检查               | 结果与边界                                                                                                                                                                                                                                                                                                                                        |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 数据转换           | 服务器 `conversion_manifest.json` 为 `complete`，记录了 50 scene / 123 episode / 28,622 帧；未在本次记录中重跑转换。                                                                                                                                                                                                                              |
| 训练与验证         | 三组均有持续更新的训练 loss 和定期验证 loss；E1/E2/E3 均已生成 checkpoint。训练进程未因本次检查而停止。                                                                                                                                                                                                                                           |
| E1 本地 checkpoint | 完整 12,500-step checkpoint 已复制到 `/home/nematrix/snap/models/DualArm_FoldTowel/experiment1/012500/`，16 个文件、约 64 MB；`rsync --checksum --dry-run` 对服务器源目录无差异。`adapter_model.safetensors` SHA256：`7557f7fb0ae2308bb1cd5088c8cbea48b1d17de87d79fd93097533aff3f18565`。部署默认 `pretrained_model` 路径指向该目录中的模型文件。 |
| E1 离线预检        | `deploy_dualarm.py --check` 通过，检查 LoRA、tokenizer、处理器统计和必要文件；仅检查结构，**没有加载 14 GB base 或做一次前向预测**。                                                                                                                                                                                                              |
| 本地部署环境       | 本机 RTX 5090 / CUDA 12.8 / BF16 小算子通过；PI05、trained RTC、PEFT、Flexiv RDK 和 RealSense 可导入；`pip check` 与 11 项部署代码单元测试通过。                                                                                                                                                                                                  |
| 真机执行           | **尚未进行**：未连接从臂、未下发动作、未测推理延迟或任务成功率。E2、E3 checkpoint 尚未复制到本地。                                                                                                                                                                                                                                                |

本地部署说明与注意事项见 [README_local_deploy.md](README_local_deploy.md)。E1 是 LoRA adapter，只有 checkpoint 不足以运行；还需匹配的 PI05 base、相机/机械臂可用状态和安全的初始姿态。base 已持久化在 `/home/nematrix/snap/models/pi05_base`，其 SHA256 与 LeRobot 官方 `pi05_base` 一致；部署入口不会再回退到 `/tmp`。

部署入口默认交互选择本地 checkpoint 及单步/连续方式；纯文件检查需传 `--check`。每次真机运行先从原始 HDF5 读取 `scene_0001` 第一个有效训练帧，通过关节距离检查后自动分段对齐；连续方式对齐后立即执行。读取状态与转换器首帧逐元素一致；LoRA 使用配置的本地 PI05 base，缺失时直接失败。

## 2026-10-23 本地部署故障修复

第一次真机启动中，采集 API 容器的三个相机进程占用了 `/dev/video0` 至 `/dev/video17`，RealSense 返回 `Device or resource busy`。停止采集容器后，相机连接通过；接着 RDK 1.9.3 被机器人软件 3.11.1 拒绝。按 [Flexiv RDK 发布记录](https://github.com/flexivrobotics/flexiv_rdk/releases)，3.11.1 对应 RDK 1.9.2，已将项目 Conda 环境、重建脚本和约束文件固定为 1.9.2。随后对左右从臂分别进行只读连接和 `info().DoF` 读取，均成功且返回 7；未使能、切控制模式或发送动作。

旧版 E1 LoRA 离线加载耗时 65.75 秒，其中 PI05 模型构造 58.8 秒，大部分是在初始化马上被 checkpoint 覆盖的随机权重。部署加载现使用 Transformers 的 `no_init_weights()`，仍严格检查所有 checkpoint 参数，并要求所有策略参数位于 `cuda:0`。同一台 RTX 5090 上 E1 加载 7.16 秒、E2 全量加载 8.0 秒、E3 LoRA 加载 7.8 秒；E1 的 893 个参数全部在 CUDA，GPU 分配约 8.72 GiB。用全零合成图像/状态完成一次 E1 trained-RTC 离线前向，得到有限的 `[50,16]` 动作块，前向约 0.42 秒；此结果不代表真实相机闭环速度或任务效果。CUDA BF16 环境检查与 `pip check` 均通过。

采集镜像内核对到 RDK 1.9.2 和 TDK 1.6.2；部署仅需 RDK。两台从臂只读查询均返回软件 `v3.11.1`、7 自由度和 7 个关节限位。三路 RealSense 已同时在本机打开，各输出 `uint8[224,224,3]` RGB 帧后关闭。PI05 官方 base 从临时目录复制到 `/home/nematrix/snap/models/pi05_base`，模型文件 SHA256 `0eb11ca9587678c1d2ef8cf32807c29f8ce53a2bfdfc1aa4a4c96f16fca59b0f` 与 [LeRobot 官方发布文件](https://huggingface.co/lerobot/pi05_base/blob/main/model.safetensors) 一致；只读查询服务器训练路径 `/mnt/work/dakun/models/pi05_base/model.safetensors` 的 SHA256 也完全相同。E1 从本地持久路径再次严格加载成功，参数全部位于 CUDA。删除了 `/tmp` 回退。RDK 文档说明重复 `SendJointPosition` 会重新规划，故暂停时改为只发送一次当前位姿目标；虚拟硬件测试覆盖这一行为。仍未进行真机使能、首帧对齐或运动测试。

## 2026-10-23 E1 15,000 首次执行与离线核查

用户在本机选择 E1 15,000 LoRA、单步模式执行。启动日志确认权重在 RTX 5090 `cuda:0`，Flexiv RDK 1.9.2 成功连接两台从臂；对齐前与采集首帧的最大关节差为 0.033 rad。`/home/nematrix/snap/deployment_logs/DualArm_FoldTowel.jsonl` 记录 43 次单步动作：第 2 步即将两侧夹爪命令到约 0.1 m，而 `scene_0001` 对应示教约第 35/37 帧才打开；第 43 步实测关节状态与对应示教帧的最大差约 0.73 rad。用户反馈出现未见过的姿态，未报告任务成功。

检查发现当前 `dispatch_single_step()` 每次调用 `engine.reset()`，结束时 `engine.pause()` 会清空 RTC 队列；每次按 Enter 都从新的首块取第一个动作，不保持训练 RTC 的动作前缀。43 次单步约持续 22 秒，也不等于训练的 15 Hz 时间推进。此模式不能评价连续 trained RTC；在重新设计前不应继续用于多步真机测试。

用原始 HDF5 状态、最近时间戳三路 RGB 图像和转换器相同的 224x224 letterbox 预处理，对 E1 15,000 做无硬件首块推理。固定随机种子 0 时，结果如下；关节误差是预测绝对目标与对应示教动作的比较，整段为开环比较，不代表闭环 RTC 结果。

| 采集样本                    | 首动作最大关节误差 | 后续 40–50 帧最大关节误差 | 左/右夹爪首次张开：预测 / 示教 |
| --------------------------- | -----------------: | ------------------------: | ------------------------------ |
| `scene_0001` 第一段第 0 帧  |          0.031 rad |                 0.338 rad | 6/35、6/37                     |
| `scene_0001` 第一段第 30 帧 |          0.027 rad |                 0.165 rad | 6/5、6/7                       |
| `scene_0025` 第一段第 0 帧  |          0.027 rad |                 0.207 rad | 6/1、6/1                       |

对相同 `scene_0001` 首帧重采样，随机种子 0/1/2 的左右夹爪首次张开分别为 6/6、8/5、32/24 帧；对应 50 帧最大关节误差为 0.338、0.472、0.172 rad。`scene_0025` 首帧示教已张开，三次预测却分别延迟至左/右 6/6、12/7、26/24 帧。模型首步关节目标较近，但整段轨迹和夹爪时序不稳定；当前证据不足以支持直接运行真机连续模式。实时图像未记录，尚无法排除现场视觉输入差异。

2026-10-23 16:26 再次启动时，采集首帧与当前关节的最大差为 0.888 rad，超过自动对齐的 0.6 rad 限值；程序在询问 `MOVE` 前中止，未执行首帧移动或策略动作。报错现已列出所有超限关节的当前值、首帧目标值和差值，限值未改变；部署测试 23 项通过。

随后按用户要求，将后续启动的首帧自动对齐上限改为 1.0 rad；仍须输入 `MOVE`，每个关节空间中间目标最多变化 0.05 rad。该变更没有在真机上执行或验证运动路径。

随后按用户要求移除部署与独立 `goHome.py` 的 `MOVE`、`RUN` 二次输入确认。运行 `goHome.py` 会在 1.0 rad 检查通过后直接分段对齐；部署选择连续模式时对齐后直接运行策略。没有放宽距离、关节限位或超时检查；本次仅完成离线测试，未执行真机运动。

独立 `goHome.py` 后续按用户要求改为同时闭合双夹爪：在关节距离检查通过后，先下发双夹爪 0 m 的闭合指令，再开始双臂关节分段回位，最终保持闭合。部署推理入口不改变首帧夹爪目标；该调整仅通过模拟 SDK 测试，未在真机执行。

## 2026-10-23 goHome 夹爪与手动模式排查

用户报告 goHome 后一侧夹爪未闭合。约 16:55 的只读 RDK 查询中，左从臂 `Rizon4-063906` 的夹爪宽度为 0.000720 m、右从臂 `Rizon4R-062106` 为 0 m，两侧均报告停止运动；读数接近零不能证明夹指实际接触。排查发现普通动作的 1 mm 死区会跳过这两种读数到零的闭合请求。现已让 goHome 显式给两侧各发送一次闭合指令，结束后等待最多 10 秒，确认 RDK 宽度不大于 1 mm 且停止运动；超时输出两侧宽度、运动状态和力值。没有自动初始化或重新标定夹爪，也没有由助手下发真机动作。

用户提供的 17:02:55 启动日志仅连接到左从臂，15 秒后使能超时，清理时 RDK 明确报 `In Manual mode`；本次尚未连接右从臂，也未执行回位或夹爪闭合。现已增加每台从臂的序列号/operational status 日志，检测到手动模式立即说明须在 Flexiv Elements 中切到 Auto；清理时跳过该模式不支持的 Stop 调用。此修改不自动切换机器人模式。

## 2026-10-23 相机帧过期排查

用户 17:08 的部署日志报 `Stale RGB camera frame`，退出时两台从臂已切换 IDLE。助手未连接机械臂，只进行了相机/GPU 诊断：三路相机独立读取 20 秒的最大帧龄约 73–75 ms，无超过 200 ms；相机与 E1 15,000 GPU RTC 同时运行 30 秒，第一轮观察出现两路约 519 ms 的旧帧，随后无超限，共取出并丢弃 440 个动作。额外预热或限制 CPU 线程的试验仍可在第一轮观察看到约 800 ms 的旧帧，因此没有把这些措施作为修复提交。

代码原来在设备连接和首帧对齐后才构造 RTC 引擎，构造中包含预处理器/tokenizer 深复制；现将构造移到设备连接之前，避免此阶段影响已启动的相机读取线程。增加引擎构造耗时日志及各路相机的名称、序列号、帧龄报错和 JSONL 字段。200 ms 阈值保持不变；这是一项针对已观察到启动阶段停顿的修正，尚未重新做 GPU 或真机验证，不能据此排除运行中的其他停帧原因。

用户查询约 10 GB 显存占用时，GPU 诊断已结束；核查时 PID 361386 不存在，`nvidia-smi --query-compute-apps` 无计算进程输出。诊断期间所有动作均丢弃，无 RDK 连接或真机动作下发。

## 2026-10-23 17:19 E2 启动失败与相机就绪检查

用户选择 E2 12,500 全量权重、连续模式。GPU 权重加载正常；RTC 构造已提前完成（1.12 秒），两台从臂 READY，首帧最大关节误差仅 0.003 rad。程序在第一轮 `get_observation()` 便中止，尚未通知模型观测或下发策略动作：全局相机帧龄 30.9 ms、左腕 `image2 / 840412062188` 1682.8 ms、右腕 143.2 ms；两台从臂随后切换 IDLE。此日志说明前一次只提前构造引擎不足以解决启动时的旧帧，不能归因于模型前向推理慢。

移除 MOVE/RUN 后不再有人为等待，相机初始化/机器人连接之后的就绪等待原本没有实现。现加入自动检查：首帧对齐与引擎线程启动后、允许模型运行前，等待三路相机各产生一张新帧，并通过原有 200 ms 帧龄和 100 ms 跨相机偏差限制；上限 15 秒。只重试缺帧、旧帧、帧时间不齐，读取线程异常和机器人状态异常直接中止；运行中的旧帧仍立即报错。未恢复人工确认，未修改 GPU/模型参数，也未运行真机或 GPU 诊断验证本次变更。

用户随后要求首帧回位后等待 1–2 秒稳定。部署现固定等待 2 秒，再启动 RTC 线程并进行三路新帧就绪检查，通过后才允许策略执行；等待不计入 60 秒策略运行时长。仅修改启动流程，未执行真机测试。

按用户最新要求，独立 `goHome.py` 顺序更新为：检查首帧起点距离 → 双夹爪显式打开到最大开度 0.1 m 并等待到位 → 保持打开完成关节回位 → 到位后显式关闭并校验两侧读数。打开超时则不进入回位；显式打开时即使目标已等于当前最大开度，也使用开爪力方向。部署推理的首帧对齐流程不随此修改改变。本次仅做离线模拟验证，未下发真机动作。

用户反馈 17:52 的 `goHome.py` 运行中两侧夹爪实物都张开，但代码在等待到位时超时。原报错没有指出是关节还是哪侧夹爪未到位，因此已增加每秒一次的双侧关节误差、目标/实测夹爪开度、运动状态和力值日志；超时时还会列出未达标项目。18:03 用户再次运行时，左夹爪实测 99.84 mm，右夹爪在命令发出后至少四秒始终报 0 mm，双臂关节误差均约 0，等待的只有右夹爪反馈；用户按 Ctrl-C 中止，尚未执行双臂回位或最终闭合。用户确认这次右侧实物也张开，因而右侧物理动作与当前 RDK 开度反馈不一致。原始 `scene_0001/lowdim/lowdim.h5` 的 `ee_state_063906` 和 `ee_state_062106` 都曾实测约 99.85 mm，采集代码记录的是 RDK `Gripper.states().width`，与当前部署读取方式一致。保留开度到位检查，避免在反馈不可信时进入机械臂运动；部署模型的右侧夹爪观测也使用该读数，须先查明反馈问题。未由助手连接或控制真机。

用户后续提供 Flexiv Elements 故障 311016 截图：`safety board: [CAT2 Error] Violation of TCP limit(s): 00001000 (X orientation | Y orientation | linear velocity | angular velocity | force | torque | - | singularity)`，另有 `joint torque = -28.13 -52.4 -20.632 20.436 5.324 -8.344 -0.824`。按显示的字段顺序，超限项是 TCP `force`；关节力矩列表不能用于计算实际 TCP 力或设置值。当前部署控制模式为 `NRT_JOINT_POSITION`，没有调用 `Robot.SetMaxContactWrench`；该 RDK 方法仅适用于笛卡尔运动/力控制，不能修改安全板的 TCP 力停机限值。应在 Flexiv Elements 的 Safety Configuration / General Limits 核对 TCP Force/Torque 限值以及当前 Normal/Reduced 模式，先排查工具/TCP、负载、接触和力传感器零点；未更改任何安全设置。

机器人力阈值接口后续核查：公开 RDK 1.9.x/2.1 文档没有 TCP force 安全限值 setter；本机 Elements 的内嵌 protobuf 协议已离线解析，确认 `proto.robot.safety.SafetyService.SetSafetyParams` 的请求包含 `params.tcp_constraints.max_force_nm`（Normal Mode）和 `max_force_rm`（Reduced Mode）。这两个字段对应力限值，`nm` 后缀并非力矩单位。内部调用文档、会话鉴权和与真机 v3.11.1 的兼容性未验证。完整协议证据与源码索引见 [RDK_FORCE_INTERFACE_AUDIT.md](RDK_FORCE_INTERFACE_AUDIT.md)；未连接真机、未调用 RPC、未修改安全设置。

## 2026-10-23 训练停止与 full checkpoint 下载

按用户要求，在服务器 `101.132.150.51:1020` 对 E2 full `torchrun` 主进程及随后要求的 E1/E3 LoRA 进程分别发送 `SIGTERM`。复查 E2 主进程、两个 DDP worker、E1/E3 主进程均已退出。三组最终已保存的 `checkpoints/last` 分别是 E1 20,000、E2 15,000、E3 17,500 step；停止不会把保存间隔内尚未保存的更新另存为 checkpoint。三组对应的 `pretrained_model` 均已在本地，E2 15,000 的 `model.safetensors` 本地与服务器 SHA256 都是 `a7341aa80c827444bdf46f87887096836f2d934308706e405d6326495ab570ef`。E2 15,000 的 `training_state`（约 11.8 GB）已补下载到 `/home/nematrix/snap/models/DualArm_FoldTowel/experiment2/015000/training_state`；`rsync --checksum --dry-run` 逐文件比对无差异，`training_step.json` 记录 step 15,000、双卡数据并行、每卡 batch 8、BF16。整个本地 checkpoint 约 20 GB；部署入口 `--check` 识别为 full，未加载模型或连接设备。

## 2026-10-23 E2 full 恢复训练

按用户后续要求，仅恢复 E2 全量微调；E1/E3 LoRA 保持停止。服务器使用原脚本 `train_dualarm_fold_towel_rtc_full.sh`，附加 `--resume=true --config_path=/mnt/work/dakun/outputs/DualArm_FoldTowel_pi05_rtc_full_gpu5_6/checkpoints/015000/pretrained_model/train_config.json`，从完整 15,000-step checkpoint 接续模型、优化器、调度器、随机状态和数据顺序，继续原输出目录与原日志。启动日志确认 `resume=True`、全局 batch 16、50,000 总 step、epoch 9/sample 6288 的数据位置；首次数据 worker 冷启动约 7 分钟。随后进度条到 50/35,000（全局 15,050 step），首条新训练 loss 为 0.021，常规 step 约 1.9 秒。GPU 5/6 各约 41 GB、利用率约 99%；E2 launcher PID 3326917，两个 worker PID 3327225/3327226。下一保存点为 17,500 step。

## 2026-10-23 部署增加可配置关节接触力矩上限

按用户要求，在 `deploy_config.json` 的左右臂配置中分别增加 `max_contact_torque_nm`，默认 `null` 不下发设置。启用时须同时把 `hardware.control_mode` 改为 `NRT_JOINT_IMPEDANCE`，每侧填写 A1–A7 共 7 个有限非负数（Nm）。程序先检查配置及控制模式，再连接 SDK；每侧使能前检查值不超过其 `RobotInfo.tau_max`，切换模式后、该侧第一次关节目标下发前调用 `Robot.SetMaxContactTorque`。数值超范围或 SDK 拒绝设置则中止。该配置由部署与 goHome 共用，修改后重启程序应用。

这是关节阻抗控制器接触力矩参数，不修改安全板 TCP force 阈值，也不能作为此前 311016 TCP force 超限的修复。默认控制模式仍为位置模式；未预填或提高真机力矩限值。Conda 部署环境运行 `python -m unittest discover -s examples/flexiv -p test_local_deploy.py`：49 项纯离线/模拟 SDK 测试通过，新增 6 项测试覆盖参数格式与模式、默认不调用、左右臂传值及调用顺序、设备占用前拒绝无效模式、额定上限和设置失败清理。未连接或运行真机。

用户随后把七个 50 写入了 `hardware.control_mode` 字符串；离线校验发现模式无效。现已修正为 `NRT_JOINT_IMPEDANCE`，并把用户填写的 `[50, 50, 50, 50, 50, 50, 50]` 移到左右臂各自的 `max_contact_torque_nm`。格式与模式校验通过；未读取真机 `tau_max`，不能确认 50 Nm 对所有关节合法或合适，连接时仍逐关节检查额定范围。模拟测试使用独立的控制模式/力矩配置，避免用户修改实际部署配置导致假机器人额定范围与测试场景混淆。未连接设备或运行真机。

20:25 用户运行 E2 15,000 连续部署：策略正常加载到 RTX 5090，RTC 构造完成；连接左臂后，`RobotInfo.tau_max` 检查明确发现 A5/A6/A7 的额定上限均为 39 Nm，配置 50 Nm 被拒绝。异常发生在该臂 `Enable` 和首次关节目标之前，尚未连接右臂或下发策略动作。现将左右臂配置都降为 `[50, 50, 50, 50, 39, 39, 39]` Nm；右臂的额定范围尚未读取，仍保留连接时逐项检查，不自动截断或跳过检查。修正后格式与控制模式离线校验通过；助手未连接设备或启动部署。

用户继续要求核查 39 Nm 能否通过代码设置。进一步检查本机 SDK 绑定、C++ 导出符号、官方头文件及 Elements 内部协议：公开 `Safety.SetJointOutputTorqueRegulator` 可通过 `tau_o_max=tau_max*limiting_factor` 调整总输出力矩调节上限，但不是 `RobotInfo.tau_max` 的直接 setter；`SetMaxContactTorque` 文档仍限定 `[0,tau_max]`。内部 `SetSafetyParamsRequest.params.joint_constraints.max_tau`（repeated float，field 5）确实存在，但与 `RobotInfo.tau_max` 的关系、写入范围和实际权限尚未验证。对独立 `RobotInfo()` 的本地赋值测试成功；`Robot.info()` 返回副本，修改该副本不能设置控制器。官方把 `tau_max` 称为软件上限，前述“额定上限”措辞已澄清。详细证据见 `RDK_FORCE_INTERFACE_AUDIT.md` 新增小节；未连接机械臂、未修改运行参数或真机安全设置。

## 真机反馈记录（待填写）

测试时请固定毛巾摆放、初始关节姿态、相机标定和运行参数，并注明日期与 checkpoint step，便于比较不同实验。收到反馈后在下表补充成功次数、失败模式及视频/日志位置；离线 loss 与真机成功率分开记录。

| ID / checkpoint | 测试时间   | 试验次数 / 成功次数             | 起始条件与标定版本                                          | 失败现象 / 安全停止                                     | 视频或日志                                                    | 结论                                                       |
| --------------- | ---------- | ------------------------------- | ----------------------------------------------------------- | ------------------------------------------------------- | ------------------------------------------------------------- | ---------------------------------------------------------- |
| E1 / 12,500     | 待反馈     | 待反馈                          | 待反馈                                                      | 待反馈                                                  | 待反馈                                                        | 待反馈                                                     |
| E1 / 15,000     | 2026-10-23 | 单步执行 1 次，43 步 / 0 次成功 | `scene_0001` 首帧，初始最大关节差 0.033 rad；现场图像未保存 | 第 2 步提前张开夹爪，关节逐步偏离示教；用户报告陌生姿态 | `/home/nematrix/snap/deployment_logs/DualArm_FoldTowel.jsonl` | 单步实现不符合 trained RTC；离线首块也不稳定，暂停真机测试 |
| E2 / 待选择     | 待反馈     | 待反馈                          | 待反馈                                                      | 待反馈                                                  | 待反馈                                                        | 待反馈                                                     |
| E3 / 待选择     | 待反馈     | 待反馈                          | 待反馈                                                      | 待反馈                                                  | 待反馈                                                        | 待反馈                                                     |
