# 本地 RTC 全量微调部署

这是 `examples/flexiv/` 部署代码的一份副本，和 [本地 checkpoints](../checkpoints/README.md) 放在同一 `pi05/` 目录下。来源代码提交为 `ab265aae`；复制了部署入口、checkpoint 检查、RDK/RealSense 适配、回初始姿态入口及离线测试。

副本的模型菜单适配 `checkpoints/rtc_full/<step>/pretrained_model/`，按 5 万、4.5 万、4 万、3 万步排列，`last` 指向最终的 `050000`。原 `examples/flexiv/` 入口仍可单独使用。以后修改两份部署代码时应核对差异。

```text
deployment/
├── run.sh                       # 选择本机环境和指定 checkpoint
├── deploy_dualarm.py             # 交互选择模型、单步/连续运行
├── deploy_config.json            # 默认 last；本机机械臂和相机配置
├── flexiv_deploy_contract.py     # 离线配置、权重、统计检查
├── flexiv_deploy_hardware.py     # RDK / RealSense 适配
├── goHome.py                     # 回到采集首帧姿态
└── test_local_deploy.py          # 离线 / 模拟设备测试
```

## 检查四个 checkpoint

在仓库根目录运行，`--check` 只读取模型文件，不加载模型，也不连接硬件：

```bash
deployment=src/lerobot/policies/pi05/deployment
bash "$deployment/run.sh" 050000 --check
bash "$deployment/run.sh" 045000 --check
bash "$deployment/run.sh" 040000 --check
bash "$deployment/run.sh" 030000 --check
```

从其他工作目录也可以使用 `run.sh` 的绝对路径。`LEROBOT_PYTHON` 可指定 Python；默认使用已激活环境，否则使用 `/home/nematrix/snap/.conda-envs/lerobot-deploy/bin/python`。

## 部署

不带参数会先选择模型，再选择单步或连续运行：

```bash
bash src/lerobot/policies/pi05/deployment/run.sh
```

明确选择模型和单步运行：

```bash
bash src/lerobot/policies/pi05/deployment/run.sh 050000 --run-mode step
```

上述部署命令会连接本机机械臂和相机，随后按现有流程恢复初始姿态。连续运行使用 `--run-mode continuous`。设备序列号、网络接口、力矩限制和初始姿态来源沿用现有 Flexiv 部署配置，详见 [原部署说明](../../../../../examples/flexiv/docs/README_local_deploy.md)。

`deploy_config.json` 已指向本次下载的最终权重和共享 tokenizer；全量 checkpoint 无需加载 base 权重。实际训练配置、预处理文件中的服务器路径保留原始内容，模型与 tokenizer 由部署入口显式指定本地路径。

如需调整设备，复制配置为 Git 忽略的 `deploy_config.local.json`，再传 `--config /绝对路径/deploy_config.local.json`。观测首帧仍从本机 `/home/nematrix/snap/reports/dual_arm_fold_towel_2026-10-23/conversion_manifest.json` 与其指向的采集 HDF5 读取；部署日志写入 `/home/nematrix/snap/deployment_logs/DualArm_FoldTowel_rtc_full.jsonl`。

## 离线测试

```bash
UV_CACHE_DIR=/tmp/flexivstudio-uv \
/home/nematrix/snap/.conda-envs/lerobot-deploy/bin/uv run --no-project \
  --python /home/nematrix/snap/.conda-envs/lerobot-deploy/bin/python \
  python -m unittest discover -s src/lerobot/policies/pi05/deployment -p test_local_deploy.py -v
```
