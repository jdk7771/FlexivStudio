# 部署力阈值接口核查（2026-10-23）

结论：当前部署使用的 Flexiv RDK 1.9.2 有控制器接触力/力矩调节接口，但没有公开设置安全板 TCP Force 停机限值的接口。核查覆盖部署调用链、本机 Python 类成员和文档、本机二进制导出的 C++ 符号，以及官方同版本 C++ 头文件。未连接机械臂、未修改运动或安全参数。

进一步核查确认：Elements 内部存在 `SafetyService.SetSafetyParams`，离线恢复的协议定义包含 `params.tcp_constraints.max_force_nm` 和 `max_force_rm`。具体定义及证据见文末；这不等于当前 RDK 提供了公开的调用方法。

## 实际部署调用链

1. 初次核查时 `deploy_config.json:28` 指定 `NRT_JOINT_POSITION`；用户启用可选接触力矩配置后已改为 `NRT_JOINT_IMPEDANCE`。
2. `deploy_dualarm.py:406`（连续模式）和 `deploy_dualarm.py:271`（单步模式）调用 `robot.send_action(...)`。
3. `flexiv_deploy_hardware.py:290` 的 `FlexivDualArm.send_action()` 校验关节位置和夹爪宽度，在第 296 行调用 `flexivrdk.Robot.SendJointPosition(...)`。
4. 本机 Python 包入口是 `/home/nematrix/snap/.conda-envs/lerobot-deploy/lib/python3.12/site-packages/flexivrdk/__init__.py`，版本为 `1.9.2`，通过 `from .flexivrdk import *` 导入编译扩展。
5. 实际 C++ 实现在同目录 `flexivrdk.cpython-312-x86_64-linux-gnu.so` 中。本机包没有对应实现 `.cpp`；本次使用 `nm -D -C` 核对了导出的接口符号。

部署现已增加可选的关节接触力矩配置：`hardware.arms.left/right.max_contact_torque_nm`。默认 `null` 不调用设置；填写各关节数值时，必须同时把 `hardware.control_mode` 改为 `NRT_JOINT_IMPEDANCE`，连接后、该臂首次下发关节目标前调用 `SetMaxContactTorque`。`read_state()` 仍仅使用关节位置和夹爪宽度，未加入应用层机械臂力阈值检查。夹爪 `Move(..., force)` 的开合力与机械臂 TCP 力限值无关。

## 找到的相关接口

| 接口                                                                     | 参数和用途                                                               | 当前部署可否直接使用                                                    |
| ------------------------------------------------------------------------ | ------------------------------------------------------------------------ | ----------------------------------------------------------------------- |
| `Robot.SetMaxContactWrench(max_wrench)`                                  | 六维 `[Fx,Fy,Fz,Mx,My,Mz]`，单位 N/Nm；调节笛卡尔运动控制器的接触力/力矩 | 不可；要求笛卡尔运动/力控制或 super primitive 模式                      |
| `Robot.SetMaxContactTorque(max_torques)`                                 | 每关节接触力矩，单位 Nm；调节关节阻抗控制器                              | 已接入可选配置；要求 `NRT_JOINT_IMPEDANCE`，默认不调用                  |
| `Safety.SetJointOutputTorqueRegulator(limiting_factor, error_threshold)` | 调节总输出力矩上限系数及持续超限周期数                                   | 可在 IDLE 中配置，但不是 TCP 力安全限值，不应当作本次 311016 的设置接口 |

前两个接口均在官方 `robot.hpp` 中声明；本机 Python 绑定和 C++ 导出符号中均存在。它们调节控制器输出，不会提高安全板的 TCP Force 停机阈值。

`Safety` 的公开设置方法只有 `SetJointPositionLimits`、`SetJointVelocityNormalLimits`、`SetJointVelocityReducedLimits` 和 `SetJointOutputTorqueRegulator`。`SafetyLimits` 只有 `q_min`、`q_max`、`dq_max_normal`、`dq_max_reduced`；没有 TCP 力限值字段。官方同版本 `safety.hpp` 与本机 Python/二进制核查一致，未发现 C++ 有而 Python 没绑定的 TCP 力限值设置接口。

## 可以读取力，但不能据此推断安全板阈值

`Robot.states()` 暴露 `ext_wrench_in_tcp`、`ext_wrench_in_world`、`tau_ext` 等反馈。可以另加应用层力监测及主动停止，但这与修改安全板阈值是不同功能；这些反馈也不意味着能读出本次故障实际触发时的安全板检测值。

## 官方同版本代码位置

- [robot.hpp：SetMaxContactTorque，第 533 行](https://github.com/flexivrobotics/flexiv_rdk/blob/v1.9.2/include/flexiv/rdk/robot.hpp#L533)
- [robot.hpp：SetMaxContactWrench，第 693 行](https://github.com/flexivrobotics/flexiv_rdk/blob/v1.9.2/include/flexiv/rdk/robot.hpp#L693)
- [safety.hpp：Safety 全部公开接口](https://github.com/flexivrobotics/flexiv_rdk/blob/v1.9.2/include/flexiv/rdk/safety.hpp)
- [safety.hpp：SetJointOutputTorqueRegulator，第 142 行](https://github.com/flexivrobotics/flexiv_rdk/blob/v1.9.2/include/flexiv/rdk/safety.hpp#L142)

本次 311016 是用户截图所示的安全板 TCP force 超限。当前公开 SDK 无对应的阈值 setter，需在 Flexiv Elements 的安全配置中核对，不能通过修改采集端 `rizon_teleop.py` 影响部署。

## 补充：找到 Elements 内部安全参数设置入口

继续检查本机 `/home/nematrix/FlexivElementsStudio/FlexivElementsStudio` 的导出符号，实际发现：

```text
0000000003cb63c0 T fvr::RobotSafetyClient::SetSafetyParams(proto::SafetyParams*)
0000000003cb6080 T fvr::RobotSafetyClient::GetSafetyParamsApplied(proto::SafetyParams*)
0000000003cb6220 T fvr::RobotSafetyClient::GetSafetyParamsBoundary(proto::SafetyParams*)
0000000004022bd0 T proto::robot::safety::SafetyService::Stub::SetSafetyParams(grpc_impl::ClientContext*, proto::SetSafetyParamsRequest const&, proto::FvrReturnValue*)
```

该可执行文件和同目录 `RobotControlApp` 的字符串中均包含内部 RPC 名称：

```text
/proto.robot.safety.SafetyService/SetSafetyParams
```

本机 RDK 1.9.2 的 `.so` 导出符号中没有 `SetSafetyParams`。因此准确说法是：安全参数有 Elements 内部设置入口；当前公开 RDK 未暴露 TCP 力安全限值的设置接口。最初符号核查不能确定消息结构，后续通过离线协议恢复确认了具体字段（见下）；权限校验和与真机软件版本的兼容性仍未验证，本次未调用内部服务。

本机模拟器安全配置模板提供了对应字段的直接证据：

`/home/nematrix/FlexivElementsStudio/user_data_ui/simDir/simulator0/A02L-00-M6-KPKSz8/safety_functions.template.xml`

| 行号 | 字段                     | 含义                      |
| ---- | ------------------------ | ------------------------- |
| 95   | `max_normal_tcp_force`   | 正常模式 TCP 力安全限值   |
| 96   | `max_normal_tcp_torque`  | 正常模式 TCP 力矩安全限值 |
| 101  | `max_reduced_tcp_force`  | 减速模式 TCP 力安全限值   |
| 102  | `max_reduced_tcp_torque` | 减速模式 TCP 力矩安全限值 |

这些是模拟器模板字段，不代表两台真机实际生效的设置；修改该模板不会设置真机阈值。后续已确认内部协议包含 TCP force 字段，但未验证实际配置写入流程。

另核对了官方 RDK 2.1 的 [Safety 公开接口](https://www.flexiv.com/software/rdk/api/v2.1/classflexiv_1_1rdk_1_1_safety.html)，同样未列出 TCP force 安全限值 setter；不能据此建议为当前 v3.11.1 真机升级 RDK。

## 文档检索与内部协议字段确认

官方 [RDK 1.9 Safety API](https://www.flexiv.com/software/rdk/api/classflexiv_1_1rdk_1_1_safety.html)、[RDK 2.1 Safety API](https://www.flexiv.com/software/rdk/api/v2.1/classflexiv_1_1rdk_1_1_safety.html) 和 `v1.9.2` 仓库全部公开头文件已核对；没有找到 TCP force 安全阈值的公开 setter。官方 [API Overview](https://www.flexiv.com/software/rdk/manual/v1.x/api_overview.html) 标注 Python 对公开 `Safety` API 为完整支持，因此没有证据表明此功能只是 Python 漏绑。对 `SetSafetyParams`、`RobotSafetyClient`、`GetSafetyParamsApplied` 的公开文档检索没有找到厂商调用说明；不能据此断言厂商内部不存在文档。

离线读取本机 Elements ELF 文件中导出的 protobuf `DescriptorTable`，恢复并成功解析 `RobotServiceSafety.proto` 和 `RobotConfig.proto` 的 `FileDescriptorProto`。未运行源二进制程序，未连接机器人。恢复的协议定义证实：

```text
service: proto.robot.safety.SafetyService
method: SetSafetyParams
request: proto.SetSafetyParamsRequest
response: proto.FvrReturnValue

SetSafetyParamsRequest.serial_number                 string, field 1
SetSafetyParamsRequest.params                        SafetyParams, field 2
SafetyParams.tcp_constraints                         TCPConstraints, field 5
TCPConstraints.max_force_nm                          float, field 12
TCPConstraints.max_force_rm                          float, field 16
```

结合正常/减速模式的 XML 字段命名，`nm` 对应 Normal Mode，`rm` 对应 Reduced Mode；此处后缀 `nm` 不是力矩单位 Nm，两个 `max_force` 字段均表示力限值。完整字段路径分别为 `request.params.tcp_constraints.max_force_nm` 和 `request.params.tcp_constraints.max_force_rm`。

同一协议还包含 `GetSafetyParamsApplied` 和 `GetSafetyParamsBoundary`，输入 `proto.GetSafetyParamsRequest`，输出 `proto.GetSafetyParamsResponse`。它们的存在说明协议区分已应用设置和参数边界，但本次没有调用以读取真机值。

证据文件保存在当前目录的 `sdk_reference/elements_safety_protocol/`：

- [RobotServiceSafety.descriptor.txt](sdk_reference/elements_safety_protocol/RobotServiceSafety.descriptor.txt)：服务和方法定义。
- [RobotConfig.descriptor.txt](sdk_reference/elements_safety_protocol/RobotConfig.descriptor.txt)：请求结构及 TCP force 字段。
- [provenance.json](sdk_reference/elements_safety_protocol/provenance.json)：源文件路径、SHA256、提取方式和验证范围。

这是本机标记为 RobotControlApp v3.11.2 的 Elements 包中的内部协议证据；不保证用户真机 v3.11.1 的接口兼容性。尚未确认会话鉴权、允许范围、持久化及生效步骤，不应把协议字段存在等同于可以直接通过当前 RDK 设置成功。未调用此内部接口或修改安全阈值；部署新增的可选关节接触力矩配置使用公开 RDK 方法，与此内部 RPC 无关。

## 39 Nm 与关节力矩设置入口的进一步核查

用户运行日志确认左臂 `RobotInfo.tau_max` 的 A5/A6/A7 为 39 Nm。官方 `data.hpp` 称 `tau_max` 为关节力矩**软件上限**；前面记录中的“额定上限”表述不够准确。当前没有证据确定该值是出厂值、维护设置还是与某项安全参数联动。

本次直接检查部署 Conda 环境 RDK 1.9.2 的类成员和 docstring，并核对 `.so` 导出符号及官方头文件：

| 入口                                                                     | 已确认的行为                                                                                                                            | 与 39 Nm 的关系                                                                                |
| ------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `Robot.SetMaxContactTorque(max_torques)`                                 | 阻抗控制的每关节接触力矩调节值，公开规定范围 `[0, RobotInfo.tau_max]`                                                                   | 当前对应关节只能填到 39 Nm                                                                     |
| `Safety.SetJointOutputTorqueRegulator(limiting_factor, error_threshold)` | IDLE 下设置总输出力矩调节器；上限公式 `tau_o_max = tau_max * limiting_factor`，连续超限超过指定周期后触发 minor fault；需要 Safety 密码 | 能改总输出调节上限，文档未说明会修改 `RobotInfo.tau_max` 或扩展 `SetMaxContactTorque` 允许范围 |
| `SafetyService.SetSafetyParams` 的 `params.joint_constraints.max_tau`    | Elements 内部请求消息包含此关节约束字段                                                                                                 | 有字段及设置 RPC 的离线证据，但尚未验证是否可写、允许范围，或它是否决定 `RobotInfo.tau_max`    |

力矩调节器接口的默认参数是 `limiting_factor=1.3, error_threshold=50`；按公式 39 × 1.3 = 50.7 Nm，只表示使用这一系数计算的总输出调节值，不能推断当前机器人已经应用这一设置，也不能把它当作 TCP force 的安全板阈值。

内部协议字段的完整路径及编号为：`SetSafetyParamsRequest.params`（field 2）→ `SafetyParams.joint_constraints`（field 4）→ `JointConstraints.max_tau`（field 5，repeated float）。证据见同目录 `sdk_reference/elements_safety_protocol/RobotConfig.descriptor.txt` 与 `RobotServiceSafety.descriptor.txt`。模拟器模板还含 `link1_torque_limit` 至 `link7_torque_limit`，但模板中的值不能视为真机当前设置，不能据此认定与 `RobotInfo.tau_max` 相同。

另验证 `RobotInfo.tau_max` 的 Python 属性确实可赋值，但 `Robot.info()` 明确返回 **value copy**。仅构造独立的 `RobotInfo()` 数据对象、将其字段设为七个 50 可以成功；这只修改本地数据，不向控制器写入。未构造 `Robot` 或 `Safety` 实例，未连接机械臂，未调用任何力矩/安全设置方法。当前公开绑定和导出接口中没有找到直接设置 `RobotInfo.tau_max` 的 setter，内部 `max_tau` 的实际关联仍待验证。

官方依据：[Robot.info 与 SetMaxContactTorque](https://github.com/flexivrobotics/flexiv_rdk/blob/v1.9.2/include/flexiv/rdk/robot.hpp)、[Safety.SetJointOutputTorqueRegulator](https://www.flexiv.com/software/rdk/api/classflexiv_1_1rdk_1_1_safety.html)、[RobotInfo 字段定义](https://github.com/flexivrobotics/flexiv_rdk/blob/v1.9.2/include/flexiv/rdk/data.hpp)。
