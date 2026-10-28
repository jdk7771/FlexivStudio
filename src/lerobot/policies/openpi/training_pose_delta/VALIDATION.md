# 验证记录（2026-10-27）

- `--test`：10 passed。验证 base/TCP 两种坐标系的非交换旋转和反变换、
  chunk 共同原点、左右臂独立性、四元数正负等价与 ±π 附近连续性、
  SLERP 与越界拒绝、episode 末尾 padding、实际转换流程、仅训练集统计、
  14D 动作和 20D 状态处理、RTC 前缀与 padding loss 排除。
- 缩小宽度/层数的官方 `PI0Pytorch` 在 CPU 上前向反向通过，启用梯度检查点。
  视觉编码器、语言模型、动作输入投影均得到有限梯度，全部参数 `requires_grad=True`。
- 真实数据准备完成：50 scenes、123 episodes、28,622 frames；
  112 train / 11 eval；state 统计用 25,957 个训练帧，
  action 统计用 1,297,850 个训练 anchor × chunk offset 样本。
- 新 parquet 总大小约 5.3 MB；视频复用符号链接，无视频重新编码。
- 原始夹爪重新插值与既有对齐标签最大差约 2.133e-6 米，
  在 5e-6 米检查容差内；最终保存的夹爪值与原数据完全一致。
- `--check-data`：真实三路视频、tokenizer、训练/验证样本、813 个预训练模型
  state-dict 键形状检查通过。动作 shape `[2,50,32]`，实际维度 14；
  state 实际维度 20；`cuda_initialized=false`。
- pose 归一化文件 SHA256：
  `3bfb824c0e837fae5c22c3b510531d9e8c8169f2ab6e1da639e89c1c0d9f58bc`。
- bash 语法、Python AST 和启动命令预览检查通过，默认独立端口 29525。
- 当前 joint 训练的 launcher/四个 worker PID 保持运行，复查日志已到 step 17500；
  本次开始时记录的原 `training/` Python/bash/config 和官方模型 Python 文件
  SHA256 全部一致。此次更改仅新增 `training_pose_delta/` 代码和独立 pose 数据目录。

未启动 pose 全尺寸 GPU 训练，未执行机器人控制；四卡显存/速度及模型效果尚未验证。
