# FlexivStudio：训练与部署共用代码

本仓库基于 LeRobot，代码目录名保留 `lerobot`，Python 包名也仍是 `lerobot`。

| 用途                | 代码位置                                            | 使用说明                                              |
| ------------------- | --------------------------------------------------- | ----------------------------------------------------- |
| 服务器训练          | `/mnt/work/dakun/lerobot`                           | [README_server.md](README_server.md)                  |
| 本地部署            | `/home/nematrix/snap/lerobot`                       | [README_local_deploy.md](README_local_deploy.md)      |
| 原始数据转换        | `examples/port_datasets/port_flexiv.py`             | [README_flexiv.md](../port_datasets/README_flexiv.md) |
| 训练/部署契约与检查 | `flexiv_deploy_contract.py`、`test_local_deploy.py` | [对齐记录](SOURCE_ALIGNMENT.md)                       |

服务器与本地使用同一套 `src/lerobot`。训练环境按服务器驱动使用 CUDA 11.8，本地 RTX 5090 部署环境使用 CUDA 12.8；依赖约束分别保存在 `server-constraints.txt` 和 `local-deploy-constraints.txt`。

## Git 同步

新机器保留 `lerobot` 目录名克隆：

```bash
git clone https://github.com/jdk7771/FlexivStudio.git lerobot
cd lerobot
git remote add upstream https://github.com/huggingface/lerobot.git
```

已有目录完成首次对齐后，在服务器或本地更新：

```bash
cd /你的路径/lerobot
git status --short
git pull --ff-only origin main
git rev-parse HEAD
```

两端 `HEAD` 相同且 `git status --short` 为空，表示受 Git 管理的代码一致。有本地修改时先提交或暂存，再更新。不要用 `rsync --delete` 或 `git reset --hard` 覆盖另一端的工作。

修改后在对应机器提交，再由另一端拉取：

```bash
git add 具体修改的文件
git commit -m "Describe the change"
git push origin main
```

`origin` 指向 FlexivStudio，`upstream` 指向 Hugging Face LeRobot。上游更新需要单独合并并验证，日常两端同步使用 `origin/main`。GitHub 写入凭据通过凭据管理器或 SSH 配置，不放进仓库 URL、脚本或配置文件。

## 数据与模型

Git 管理源码、配置模板、依赖约束和文档。训练数据、模型、优化器状态、部署日志保留在各机器的数据目录；代码更新后原路径继续有效。`deploy_config.json` 记录本地设备与路径，其他机器需要按实际硬件调整，可复制为 Git 忽略的 `deploy_config.local.json` 后通过 `--config` 使用。

两端统一动作顺序为左臂 7 关节、左夹爪、右臂 7 关节、右夹爪；3 路 224×224 RGB，15 Hz，chunk 50，关节相对动作、夹爪绝对动作。部署前使用 checkpoint 的预处理/后处理和本地检查入口，详见部署说明。

## 无硬件验证

在已安装对应依赖的环境中执行：

```bash
python -m unittest discover -s examples/port_datasets -p test_port_flexiv.py -v
python -m unittest discover -s examples/flexiv -p test_local_deploy.py -v
for script in examples/flexiv/*.sh; do bash -n "$script"; done
```

这些单元测试使用临时数据或模拟设备，不连接机械臂或相机。训练脚本可使用 `--print-command` 检查参数。
