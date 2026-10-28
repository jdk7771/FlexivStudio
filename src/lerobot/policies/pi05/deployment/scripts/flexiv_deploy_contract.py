"""Offline deployment contract checks, using only the Python standard library."""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path

ACTION_NAMES = [
    name
    for side in ("left", "right")
    for name in (*[f"{side}_joint_{i}" for i in range(1, 8)], f"{side}_gripper")
]
CAMERA_NAMES = ("image", "image2", "image3")


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def tensor_header(path: Path) -> dict:
    """Inspect tensor names and truncation without importing torch or reading tensor data."""
    with path.open("rb") as stream:
        size_bytes = stream.read(8)
        if len(size_bytes) != 8:
            raise ValueError(f"Invalid safetensors file: {path}")
        header_size = struct.unpack("<Q", size_bytes)[0]
        if not 0 < header_size <= 64 * 1024 * 1024:
            raise ValueError(f"Invalid safetensors header size: {path}")
        header = json.loads(stream.read(header_size))
    tensors = {name: info for name, info in header.items() if name != "__metadata__"}
    if (
        not tensors
        or max(info["data_offsets"][1] for info in tensors.values()) + header_size + 8 != path.stat().st_size
    ):
        raise ValueError(f"Empty, incomplete, or invalid safetensors file: {path}")
    return tensors


def contact_torque_limits(hw: dict, side: str) -> list[float] | None:
    """Validate optional joint contact torque settings without importing the SDK."""
    limits = hw["arms"][side].get("max_contact_torque_nm")
    if limits is None:
        return None
    if hw["control_mode"] != "NRT_JOINT_IMPEDANCE":
        raise ValueError(
            f"hardware.arms.{side}.max_contact_torque_nm requires hardware.control_mode=NRT_JOINT_IMPEDANCE"
        )
    if (
        not isinstance(limits, list)
        or len(limits) != 7
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            for value in limits
        )
    ):
        raise ValueError(
            f"hardware.arms.{side}.max_contact_torque_nm must contain seven finite "
            "non-negative numbers (A1-A7, Nm), or be null"
        )
    return [float(value) for value in limits]


def validate_settings(cfg: dict, *, rtc: bool = True) -> None:
    if cfg["fps"] != 15:
        raise ValueError(
            "These checkpoints use 15 Hz training data; changing playback speed changes the task"
        )
    integer_settings = ["chunk_size", "num_inference_steps"]
    if rtc:
        integer_settings.extend(("training_max_delay", "rtc_execution_horizon", "rtc_queue_threshold"))
    for name in integer_settings:
        if type(cfg[name]) is not int or cfg[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if rtc:
        delay, chunk, horizon = cfg["training_max_delay"], cfg["chunk_size"], cfg["rtc_execution_horizon"]
        if not delay <= horizon <= chunk - delay or not delay <= cfg["rtc_queue_threshold"] < chunk:
            raise ValueError("RTC prefix capacity and queue threshold do not fit the trained delay/chunk")
    for name in ("max_observation_age_s", "action_timeout_s", "duration_s"):
        if not math.isfinite(cfg[name]) or cfg[name] <= 0:
            raise ValueError(f"{name} must be finite and positive")
    start = cfg["start_pose"]
    if not start["scene"].startswith("scene_") or not start["manifest"]:
        raise ValueError("start_pose must name a collection scene and manifest")
    for name in (
        "max_joint_delta_rad",
        "waypoint_step_rad",
        "joint_tolerance_rad",
        "gripper_tolerance_m",
        "timeout_s",
    ):
        if not math.isfinite(start[name]) or start[name] <= 0:
            raise ValueError(f"start_pose.{name} must be finite and positive")
    if not 0 < start["joint_tolerance_rad"] < start["waypoint_step_rad"] <= 0.1:
        raise ValueError("Initial-pose waypoint/tolerance settings are unsafe")
    if start["max_joint_delta_rad"] > 1.0 or start["gripper_tolerance_m"] > 0.01:
        raise ValueError("Initial-pose movement exceeds the configured safety bounds")
    hw = cfg["hardware"]
    if list(hw["arms"]) != ["left", "right"] or set(hw["cameras"]) != set(CAMERA_NAMES):
        raise ValueError("Expected left/right followers and image/image2/image3 cameras")
    serials = [arm["serial"] for arm in hw["arms"].values()]
    if len(set(serials)) != 2 or len(set(hw["cameras"].values())) != 3:
        raise ValueError("Robot and camera serial numbers must be distinct")
    if hw["control_mode"] not in ("NRT_JOINT_POSITION", "NRT_JOINT_IMPEDANCE"):
        raise ValueError("SendJointPosition requires an NRT joint position or impedance mode")
    for side in ("left", "right"):
        contact_torque_limits(hw, side)
    if hw["camera_fps"] != 15 or (hw["camera_width"], hw["camera_height"]) != (1280, 720):
        raise ValueError("Use the same RGB resolution and rate as the recorded data: 1280x720 at 15 Hz")
    for name in (
        "connect_timeout_s",
        "max_joint_velocity_rad_s",
        "max_joint_acceleration_rad_s2",
        "max_target_error_rad",
        "gripper_speed_m_s",
        "gripper_deadband_m",
        "max_camera_age_s",
        "max_camera_skew_s",
    ):
        if not math.isfinite(hw[name]) or hw[name] <= 0:
            raise ValueError(f"hardware.{name} must be finite and positive")
    if not 0 <= hw["gripper_min_m"] < hw["gripper_max_m"] <= 0.1:
        raise ValueError("Expected gripper width in meters, within [0, 0.1]")
    if not hw["gripper_open_force_n"] < 0 < hw["gripper_close_force_n"]:
        raise ValueError("Use collection's signed gripper forces: negative open, positive close")


def inspect_checkpoint(cfg: dict, *, rtc: bool = True) -> dict:
    """Reject incompatible/incomplete checkpoints BEFORE model or hardware creation."""
    checkpoint = Path(cfg["checkpoint"])
    model = read_json(checkpoint / "config.json")
    training_delay = model.get("rtc_training_max_delay", 0)
    if type(training_delay) is not int or training_delay < 0:
        raise ValueError("Checkpoint rtc_training_max_delay must be a non-negative integer")
    validate_settings(cfg, rtc=rtc)
    expected = {
        "type": "pi05",
        "chunk_size": cfg["chunk_size"],
        "use_relative_actions": True,
    }
    # Training with a prefix also supports prefix-free synchronous inference.
    # Only trained-RTC execution requires matching the checkpoint's delay capacity.
    if rtc:
        expected["rtc_training_max_delay"] = cfg["training_max_delay"]
    for key, value in expected.items():
        if model.get(key) != value:
            raise ValueError(f"Checkpoint {key}={model.get(key)!r}, expected {value!r}")
    execution_steps = model.get("n_action_steps", model["chunk_size"])
    if type(execution_steps) is not int or not 0 < execution_steps <= model["chunk_size"]:
        raise ValueError("Checkpoint n_action_steps must be within the predicted chunk")
    if not rtc:
        requested_steps = cfg.get("n_action_steps", execution_steps)
        if type(requested_steps) is not int or not 1 <= requested_steps <= model["chunk_size"]:
            raise ValueError(f"n_action_steps must be an integer between 1 and {model['chunk_size']}")
    if (
        model.get("n_obs_steps", 1) != 1
        or model.get("use_visual_memory")
        or model.get("use_proprioceptive_memory")
    ):
        raise ValueError("This deployment expects a current-observation PI05 checkpoint")
    names = model.get("action_feature_names")
    if names is not None and names != ACTION_NAMES:
        raise ValueError("Checkpoint's action order differs from the dataset converter")
    if model.get("relative_exclude_joints") != ["gripper"]:
        raise ValueError("Checkpoint must use Joint Delta + Gripper Absolute")
    inputs = model["input_features"]
    if inputs["observation.state"]["shape"] != [16] or model["output_features"]["action"]["shape"] != [16]:
        raise ValueError("Checkpoint must consume/produce 16 dimensions")
    image_keys = {key for key in inputs if key.startswith("observation.images.")}
    if image_keys != {f"observation.images.{name}" for name in CAMERA_NAMES}:
        raise ValueError("Checkpoint RGB camera keys differ from the converter")
    for key in image_keys:
        if inputs[key]["shape"] != [3, 224, 224]:
            raise ValueError(f"Unexpected image shape for {key}")
    for filename, normalizer in (
        ("policy_preprocessor.json", "normalizer_processor"),
        ("policy_postprocessor.json", "unnormalizer_processor"),
    ):
        steps = read_json(checkpoint / filename)["steps"]
        for step in steps:
            if "state_file" in step:
                tensor_header(checkpoint / step["state_file"])
            for artifact in step.get("artifacts", {}).values():
                if not (checkpoint / artifact).exists():
                    raise ValueError(f"Missing checkpoint processor artifact: {artifact}")
        stats = next((step for step in steps if step.get("registry_name") == normalizer), None)
        if not stats or not stats.get("state_file"):
            raise ValueError(f"Missing trained normalization statistics in {filename}")
        header = tensor_header(checkpoint / stats["state_file"])
        features = ("observation.state", "action") if filename == "policy_preprocessor.json" else ("action",)
        for feature in features:
            for quantile in ("q01", "q99"):
                if header.get(f"{feature}.{quantile}", {}).get("shape") != [16]:
                    raise ValueError(f"Missing 16D {feature}.{quantile} statistics in {filename}")
        if filename == "policy_preprocessor.json":
            relative = next(s for s in steps if s.get("registry_name") == "relative_actions_processor")[
                "config"
            ]
            if not relative.get("enabled") or relative.get("exclude_joints") != ["gripper"]:
                raise ValueError("Saved preprocessor does not match relative joints / absolute grippers")
            if relative.get("action_names") not in (None, ACTION_NAMES):
                raise ValueError("Saved preprocessor has a different action order")
        else:
            absolute = next(s for s in steps if s.get("registry_name") == "absolute_actions_processor")[
                "config"
            ]
            if not absolute.get("enabled"):
                raise ValueError("Saved postprocessor must restore absolute joint targets")
    tokenizer = Path(cfg["tokenizer"])
    if not (tokenizer / "tokenizer_config.json").is_file() or not any(
        (tokenizer / name).is_file() for name in ("tokenizer.json", "tokenizer.model")
    ):
        raise ValueError("Missing local PaliGemma tokenizer files")
    adapter = (checkpoint / "adapter_config.json").is_file()
    if bool(model.get("use_peft")) != adapter:
        raise ValueError("Checkpoint use_peft flag and adapter files disagree")
    tensor_header(checkpoint / ("adapter_model.safetensors" if adapter else "model.safetensors"))
    if adapter:
        tensor_header(Path(cfg["base_model"]) / "model.safetensors")
    return {
        "checkpoint": str(checkpoint),
        "kind": "LoRA" if adapter else "full",
        "model_config": model,
    }
