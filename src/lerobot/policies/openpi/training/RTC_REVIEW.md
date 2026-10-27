# RTC 复查（2026-10-27）

结论：当前训练入口的 RTC 训练语义与本地 LeRobot pi05 一致。本次仅添加对照测试和说明，
没有修改、重启或中断正在运行的训练代码。模型本体仍为官方 openpi `PI0Pytorch`。

## 直接对照的来源

相对于 policies 目录：

- `pi05/modeling_pi05.py`：`_sample_training_rtc_prefix_mask`、`_reduce_training_rtc_loss`、
  `PI05Policy.forward`、模型 `forward` / `embed_suffix`。
- `common/flow_matching.py`：`make_flow_matching_inputs`，`FlowConvention.NOISE_AT_ONE`。
- `common/vla_utils.py`：正弦时间编码与 attention mask。
- `pi_gemma.py`：`PiGemmaRMSNorm` 的逐动作时间条件。

`test_rtc_reference.py` 从这些本地文件提取原函数/类 AST 并直接执行，避免同时导入
不兼容的 transformers 4 与 5；没有另写一套“参考公式”来证明自身正确。

## 核对结果

| 环节 | LeRobot / 本次 openpi |
| --- | --- |
| 前缀长度 | 每个样本独立均匀采样 0…8，包含 8 |
| 前缀输入 | 干净的真实动作，模型时间为 0 |
| 后缀输入 | `x_t = t * noise + (1-t) * actions` |
| 监督目标 | `noise - actions` |
| 时间条件 | 每个动作一个时间，通过正弦编码 → time MLP → AdaRMS |
| attention | 沿用双向 action block；RTC 不增加 causal/prefix attention mask |
| loss | 只取真实 16 维，去掉干净前缀，除以整批有效后缀元素数 |
| DDP | 各 rank 先按本地有效元素数归约，再由 DDP 平均梯度，与参考方式相同 |
| 训练入口 | `training/train.py` 确实采样 prefix 并传给 DDP forward，然后调用 RTC loss |

17 项 CPU 测试全部通过，包含原来的 6 项和新增的 11 项：

- 相同 RNG 状态下，前缀采样结果及后续 RNG 状态与原函数完全相同。
- delay=0、1、8 以及边界 49：实际 openpi forward 的加噪输入、时间、逐元素 MSE、
  归约 loss 和投影梯度与 LeRobot 原函数对照，`rtol=0, atol=0`。
- AdaRMS 在 FP32 / BF16 激活下，输出、gate、输入梯度、时间条件梯度完全相同。
- 时间编码与 attention mask 完全相同。
- 缩小后的官方模型实际前向/反向和梯度检查点通过。
- 运行目录保存的 `train.py`、`runtime.py`、`pi0_pytorch.py` 源码快照与检查对象相同，
  `.venv` 中已安装的 Gemma patch 与源码替换文件相同。
- 原参考 JSON 的 SHA256 未变化；正在运行的 JSON 与准备的 JSON 相同。

日志：`/mnt/work/dakun/setup_artifacts/openpi-rtc-review-tests.log`。

## 不能声称完全相同的部分

1. 随机序列：LeRobot 先采样 noise、time，再采样 delay；本入口先采样 delay，模型内部再采样
   noise、time。LeRobot 的 Beta 时间采样在 CPU，官方 openpi 在目标设备。分布和 RTC 目标相同，
   但相同 seed 不代表相同 batch 噪声或逐步相同的 loss。本次保留官方采样方式。
2. 关闭 RTC 的边界：`max_delay=0` 时，LeRobot 返回 `None`；本入口返回全 False mask，
   仍经过逐动作时间分支并消耗一次 randint。训练目标相同，执行路径/RNG 不同；当前配置是 8。
3. 验证：本入口只由 rank 0 使用固定验证随机种子，参考实现会在所有 rank 上验证并平均，
   使用持续变化的 RNG。数据划分/256 个验证样本相同，验证随机序列不同。
4. 模型内部精度：保留官方 openpi 的参数精度策略；当前 LeRobot 移植版对视觉路径保留 FP32
   的范围不同。上述数值测试验证 RTC 算子，不能证明两个完整 3B 模型的训练逐位等价。
5. 上游 `scripts/train_pytorch.py` 没有改成 RTC 入口；必须使用本目录提供的 bash / `training/train.py`。
6. 本次只补 RTC **训练**。普通 `sample_actions` 尚未补上推理时的硬前缀钳制，实机异步 RTC
   推理需要另行接入，不能仅凭本次训练补丁宣称已支持完整 RTC 部署。

运行状态：复查时已观察到用户启动的四卡训练和有限的下降 loss（最近检查点为日志 step 350，
loss 约 0.141）。这是训练进度记录，不是模型质量、最终收敛或实机效果的验证。
