# π0.5 模型、训练与 Flexiv 部署

```text
pi05/
├── configuration_pi05.py         # 模型和策略参数定义
├── modeling_pi05.py              # 模型实现
├── processor_pi05.py             # 输入输出预处理
├── training/                    # 每个实验的 JSON 配置和训练 bash
├── checkpoints/
│   ├── rtc_full/                 # 新全量训练，含视觉：015000 / 020000 / 025000
│   └── rtc_frozen_vision/        # 旧视觉冻结：025000 / 030000 / 040000 / 045000 / 050000
└── deployment/                  # Flexiv 部署代码副本和本地启动入口
```

- [训练配置、启动和恢复](training/README.md)
- [RTC checkpoint 存放位置、视觉冻结状态与校验](checkpoints/README.md)
- [本地 Flexiv 模型选择、检查和部署](deployment/README.md)

RTC 默认部署 checkpoint 是 `checkpoints/rtc_full/025000/pretrained_model/`，该组 `last` 指向 `025000`。15k、20k、25k 来自服务器 `101.132.150.51:1024` 的 `outputs/rtc_full`，均已逐文件 SHA256 校验并通过本机离线推理。模型权重和优化器状态属于本机文件，由 Git 忽略；源码和说明单独管理。

旧目录原名 `rtc_full`，其保存配置实际为 `freeze_vision_encoder=true`，已统一改名为 `rtc_frozen_vision`，该组 `last` 仍指向旧版 `050000`。新组 `rtc_full` 的保存配置为 `freeze_vision_encoder=false`；两组模型独立存放，不按相同步数混用。
