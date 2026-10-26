# π0.5 模型、训练与 Flexiv 部署

```text
pi05/
├── configuration_pi05.py         # 模型和策略参数定义
├── modeling_pi05.py              # 模型实现
├── processor_pi05.py             # 输入输出预处理
├── training/                    # 每个实验的 JSON 配置和训练 bash
├── checkpoints/
│   └── rtc_full/                # 本地 030000 / 040000 / 045000 / 050000
└── deployment/                  # Flexiv 部署代码副本和本地启动入口
```

- [训练配置、启动和恢复](training/README.md)
- [RTC 全量微调 checkpoint 存放位置与校验](checkpoints/README.md)
- [本地 Flexiv 模型选择、检查和部署](deployment/README.md)

最终 checkpoint 是 `checkpoints/rtc_full/050000/pretrained_model/`，`last` 指向 `050000`。模型权重和优化器状态属于本机文件，由 Git 忽略；源码和说明单独管理。
