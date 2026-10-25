# Flexiv 双臂数据 → LeRobot / π0.5 training RTC

适配当前仓库的数据 API，输入为 `metadata.yaml`、`scene_*/lowdim/lowdim.h5` 和三路 `cam_*/color/<毫秒时间戳>.png`。
源目录只读，输出目录必须不存在；不上传数据，也不连接机械臂。

## 格式和动作含义

每帧保存三张 RGB 图像，以及 16 维 `observation.state` / `action`：

```text
左臂 joint_1..joint_7, left_gripper, 右臂 joint_1..joint_7, right_gripper
```

- 从臂序列号从 metadata 读取；这批数据左臂 `063906`，右臂 `062106`。
- 全局相机 `350122061994` → `observation.images.image`；左腕 `840412062188` → `image2`；右腕 `350122063150` → `image3`。
- 关节保留弧度；夹爪保留米，不做二值化或改成 0～1。
- `state(t)`：当前实测关节和实测夹爪宽度，按时间线性插值。
- `action(t)`：关节用 `q_measured(t + 1/fps)`，夹爪用 `ee_command(t)`，夹爪指令按最近过去值保持。

**限制：原始文件没有关节目标指令，因此这里的关节动作是“下一时刻实测位置”近似标签，不能还原真实控制器指令。**

数据文件中的关节动作保持绝对位置。开启 `policy.use_relative_actions` 后，LeRobot 在训练时对整个 chunk 减去当前观测的关节位置：

```text
delta[k] = q_measured(t + (k+1)/fps) - q_measured(t)
gripper[k] = ee_command(t + k/fps)
```

整个 chunk 使用同一个起点；不是每两个相邻动作之间做差。夹爪通过名字中的 `gripper` 被排除，始终用绝对值。
**不要再离线减一次 joint delta。** 脚本最后会调用仓库的 `recompute_stats(relative_action=True)`，让 `meta/stats.json` 匹配相对关节 / 绝对夹爪；其 chunk size 必须与训练一致。

RTC 的已知动作前缀由模型训练代码在线生成；转换时不拼历史前缀、不加 RTC 专用字段。

## 环境

在已有本仓库 LeRobot 环境中，额外安装 `h5py` 和 `pyyaml` 即可。Python 需 ≥ 3.12。
从头用仓库的 uv 环境安装：

```bash
cd /home/nematrix/snap/lerobot
uv sync --extra dataset --extra training --extra pi --extra peft
```

以下命令通过 `uv run --with h5py --with pyyaml` 补齐转换依赖；已有环境也可直接用 `python` 执行相同脚本。

## 先检查，再转换

仅检查时间戳、字段及有效片段，不写文件：

```bash
uv run --with h5py --with pyyaml python examples/port_datasets/port_flexiv.py \
  --source /media/nematrix/50AA6D45AA6D28A43/data/template_test \
  --dry-run
```

完整转换（把 `--task` 换成演示的真实任务描述；默认所有 scene 共用这个描述）：

```bash
uv run --with h5py --with pyyaml python examples/port_datasets/port_flexiv.py \
  --source /media/nematrix/50AA6D45AA6D28A43/data/template_test \
  --output /home/nematrix/snap/datasets/template_test_joint_rtc \
  --repo-id local/template_test_joint_rtc \
  --task '替换成实际任务描述' \
  --fps 15 \
  --chunk-size 50
```

只试转一条 scene，并把每个输出 episode 限制为 60 帧，可追加 `--max-scenes 1 --max-frames 60`，同时使用不同的输出目录。
默认使用 8 个线程并行解码和缩放 PNG，预取 32 帧，再直接流式编码 MP4，不经过中间 PNG 文件。
离线编码队列可容纳最长的一个 episode，避免录制接口在队列短暂拥堵时丢帧；写完会核对三路视频帧数。这批数据最长 699 帧，三路队列最多约 316 MB 图像内存。
可以用 `--workers 16 --prefetch-frames 64` 调整；更多线程不保证更快，本机实测 8 线程优于 16 线程。不需要训练显卡。

默认 RGB 最近帧误差上限 100 ms；低维数据插值缺口上限 50 ms。超过上限会切断 episode，并丢弃不足 50 帧的短段，不会把缺口两侧直接连接。
图像按比例缩放并补黑边到 224×224，存为 H.264 MP4；不使用深度图。
这批 50 个 scene 在默认参数下预检得到 **123 个连续 episode、28,622 帧（约 31.8 分钟）**；episode 增加来自缺帧切段，没有新增演示。

输出含 `data/`、`videos/`、`meta/info.json`、`meta/stats.json` 等 LeRobot v3 文件。
`meta/conversion_manifest.json` 记录源 scene → episode、时间范围、单位、相机对应关系和剔除帧数；`status=complete` 才表示转换和相对动作统计均完成。
如果途中失败，使用新的输出目录重跑；脚本不会覆盖已有目录。改变 chunk size 后需重新计算相对动作统计。
划分训练/验证集时，依据 manifest 按原始 scene 分组，避免同一条演示切出的多个 episode 同时落入两组。

## training RTC 对应参数

在你的 π0.5 训练命令中加入：

```bash
--dataset.repo_id=local/template_test_joint_rtc \
--dataset.root=/home/nematrix/snap/datasets/template_test_joint_rtc \
--policy.use_relative_actions=true \
--policy.relative_exclude_joints='["gripper"]' \
--policy.chunk_size=50 \
--policy.rtc_training_max_delay=8
```

`8` 是训练时随机已知前缀的最大长度，在 15 Hz 下约 0.53 秒，应按实际推理和通信延迟调整。
只开 `rtc_config` 不等于启用 training RTC，`rtc_training_max_delay` 必须大于 0。
LoRA、冻结视觉编码器等属于训练配置，不改变本转换格式。

## 脚本验证

```bash
python -m unittest discover -s examples/port_datasets -p test_port_flexiv.py -v
```

测试覆盖下一步关节标签、夹爪指令的因果取样、双臂维度顺序、重复时间戳、缺帧分段和图像补边。
已在本机通过 6 项测试（包括并发预取时图像与动作的时序对应），并用第一条真实 scene 试转 240 帧 / 4 个 episode。完整转换和上传信息见服务器说明；未执行模型训练。
