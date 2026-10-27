# 独立起始场景叠图工具

只使用一台 top RealSense 相机和桌面窗口。**不导入部署代码，不加载模型，不连接机械臂，不发运动命令，不改部署配置，也不会启动部署。** 默认 top 序列号 `350122061994`，RGB 1280×720、15 FPS。参考图保存无损 PNG。

在已有 `lerobot-deploy` 环境中使用，需要 `numpy`、`Pillow`、`pyrealsense2` 和 Tk 桌面支持。工具必须在有图形桌面的会话启动，普通无显示的 SSH 终端不能显示窗口。

## 第一次：保存 10 个参考场景

先通过原有流程让双臂回 home，并统一夹爪开度。退出占用相机的部署/采集程序，然后运行：

```bash
cd /home/nematrix/snap/lerobot/examples/flexiv/start_scene_alignment
python align.py --capture
```

摆好第一个测试场景，选择 `test_01`，确认画面和机械臂姿态后勾选确认框，按 Enter 或点击保存。程序自动进入下一个空编号，依次到 `test_10`。中途退出会保留已保存的参考，下次可以续采；已保存的编号不会被覆盖。可先建立少于十个场景再补齐。

每个场景保存 `top.png` 和 `reference.json`，记录相机参数、图像校验值、主机接收时间及备注。默认目录：

```text
/home/nematrix/snap/deployment_logs/start_scenes/
  test_01/
  ...
  test_10/
  checks/       # 后续每次对齐确认的独立留档
```

## 后续：叠图对齐同一场景

```bash
python align.py --compare test_01 --note "RTC 50k 第1次"
```

窗口将参考图和实时图以 50% 透明度叠加。调整毛巾直到边缘重影尽量消失；拖动透明度滑条可单独查看参考图（0）或实时图（1）。同时检查桌面固定标记，避免误把相机位置变化当成毛巾偏移。

勾选人工确认，按 Enter 保存并退出。工具会先释放相机，然后你再按原有方式启动 `run.sh` 或 `run_no_rtc.sh`，这些脚本不需要加任何新参数。退出叠图工具后，应保持毛巾、桌面、相机和机械臂初态不变；独立工具不会阻止部署流程再次回位。

`checks/<UTC时间>_test_01_<随机标识>/` 包含参考图副本、此次实际图、50% 叠图和 `alignment.json`，可用备注关联后续部署记录。未勾选确认、相机帧过期、相机序列号/分辨率/FPS 不匹配时不能完成确认。关闭窗口或 Esc 取消，不保存此次对齐结果。

## 其他命令

```bash
python align.py --compare                 # 窗口中选择已保存的场景
python align.py --list                    # 检查参考文件，不连接相机
python align.py --capture --directory /路径/另一套参考
python align.py --compare test_01 --serial 350122061994
```

同一套参考必须使用相同相机序列号、分辨率和 FPS。需要重建场景时使用新的 `--directory`，避免覆盖旧实验依据。

此版本采用**人工视觉确认**，没有自动毛巾分割或相似度合格判定；不会测量相机外参、机械臂关节或夹爪状态。相机未移动、机械臂回同一个 home、夹爪开度相同，需要操作人员自行确认。俯视重合也不能保证褶皱、厚度和摩擦完全一致。建议在桌面设置固定标记，固定光照与相机机位。

## 离线测试

```bash
python -m unittest discover -s . -p 'test_*.py'
```

测试使用模拟相机；有桌面时额外测试窗口及按键确认，不连接硬件。
