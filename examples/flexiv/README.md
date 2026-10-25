# Flexiv 双臂部署与实验资料

π0.5 训练统一放在 [policies/pi05/training](../../src/lerobot/policies/pi05/training/README.md)。本目录保留硬件部署入口，文档和环境安装分别放在 `docs/`、`setup/`。

```text
examples/flexiv/
├── deploy_dualarm.py             # π0.5 / trained RTC 部署入口
├── deploy_config.json            # 当前本地路径、机械臂和相机配置
├── flexiv_deploy_contract.py     # checkpoint 与动作约定检查
├── flexiv_deploy_hardware.py     # RDK / RealSense 适配
├── goHome.py                     # 返回采集首帧姿态
├── test_local_deploy.py          # 纯离线 / 模拟设备测试
├── setup/
│   ├── setup_local_deploy.sh
│   ├── local-deploy-constraints.txt
│   └── verify_local_deploy_env.py
└── docs/
    ├── README_local_deploy.md
    ├── EXPERIMENT_LOG.md
    ├── SOURCE_ALIGNMENT.md
    ├── RDK_FORCE_INTERFACE_AUDIT.md
    └── sdk_reference/
```

| 用途                         | 说明                                                           |
| ---------------------------- | -------------------------------------------------------------- |
| π0.5 训练配置、启动、恢复    | [训练入口](../../src/lerobot/policies/pi05/training/README.md) |
| 本地部署、模型路径、设备配置 | [部署说明](docs/README_local_deploy.md)                        |
| 原始数据转换                 | [数据转换](../port_datasets/README_flexiv.md)                  |
| 训练和真机测试记录           | [实验记录](docs/EXPERIMENT_LOG.md)                             |
| 源码来源和首次对齐           | [对齐记录](docs/SOURCE_ALIGNMENT.md)                           |
| RDK 接口证据                 | [接口核查](docs/RDK_FORCE_INTERFACE_AUDIT.md)                  |

## 本地入口

```bash
# 检查 checkpoint 文件，不连接设备
python examples/flexiv/deploy_dualarm.py --check

# 离线 / 模拟设备测试
python -m unittest discover -s examples/flexiv -p test_local_deploy.py -v

# 新机器安装部署环境
bash examples/flexiv/setup/setup_local_deploy.sh
```

`deploy_config.json` 沿用本机模型、数据和设备路径。其他机器可以复制为 Git 忽略的 `deploy_config.local.json`，调整后通过 `--config` 指定。模型权重、数据和部署日志继续保存在各机器原有目录。

动作顺序为左臂 7 关节、左夹爪、右臂 7 关节、右夹爪；3 路 224×224 RGB、15 Hz、chunk 50，关节相对动作、夹爪绝对动作。具体控制行为见部署说明。

## 两端代码同步

本地代码：`/home/nematrix/snap/lerobot`；训练服务器代码：`/mnt/work/dakun/lerobot`。两端共用 `src/lerobot`，环境分别使用本地 cu128 与服务器 cu118。

新机器克隆时保留 `lerobot` 目录名：

```bash
git clone https://github.com/jdk7771/FlexivStudio.git lerobot
cd lerobot
git remote add upstream https://github.com/huggingface/lerobot.git
```

已有目录更新：

```bash
git status --short
git pull --ff-only origin main
git rev-parse HEAD
```

`origin` 指向 FlexivStudio，`upstream` 指向 Hugging Face LeRobot。更新前处理本机修改；提交源码后由另一端拉取。两端提交相同且受 Git 管理的文件无修改，表示代码一致。模型、训练状态和数据通过各机器的数据目录管理。
