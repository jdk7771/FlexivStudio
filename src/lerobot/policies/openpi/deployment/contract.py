"""OpenPI dual-arm checkpoint validation and menus; no model/hardware imports."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import shared  # noqa: F401

# isort: split

from flexiv_deploy_contract import ACTION_NAMES, CAMERA_NAMES, read_json, tensor_header, validate_settings


def checkpoints(root: Path) -> list[Path]:
    return (
        sorted(
            (
                p
                for p in root.iterdir()
                if p.is_dir()
                and p.name.isdecimal()
                and all(
                    (p / f).is_file()
                    for f in ("model.safetensors", "train_config.json", "dataset_stats.json")
                )
            ),
            key=lambda p: int(p.name),
            reverse=True,
        )
        if root.is_dir()
        else []
    )


def select(label: str, choices: list[Any]) -> Any | None:
    if not choices:
        raise FileNotFoundError(f"{label}：没有完整权重；.download 暂存目录不会列入菜单")
    print("\n" + label)
    for i, choice in enumerate(choices, 1):
        print(f"  {i}. {choice.name if isinstance(choice, Path) else choice}")
    while True:
        try:
            answer = input("输入编号（q 退出）：").strip().lower()
        except EOFError:
            return None
        if answer in {"q", "quit"}:
            return None
        if answer.isdecimal() and 1 <= int(answer) <= len(choices):
            return choices[int(answer) - 1]
        print("请输入菜单中的编号。")


def choose_checkpoint(root: Path) -> Path | None:
    if checkpoints(root):
        return select("选择 OpenPI checkpoint（目录名为训练步数）：", checkpoints(root))
    folders = sorted(
        p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".") and checkpoints(p)
    )
    folder = select("选择 OpenPI 权重文件夹：", folders)
    return None if folder is None else select("选择 checkpoint（目录名为训练步数）：", checkpoints(folder))


def inspect(cfg: dict, *, hardware: bool = False) -> dict:
    validate_settings(cfg, rtc=False)
    for key in ("n_action_steps", "rtc_prefix_steps", "rtc_queue_threshold"):
        if type(cfg[key]) is not int or cfg[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if not 1 <= cfg["n_action_steps"] <= cfg["chunk_size"]:
        raise ValueError("n_action_steps must be within the predicted chunk")
    if not cfg["rtc_prefix_steps"] <= cfg["rtc_queue_threshold"] < cfg["chunk_size"]:
        raise ValueError("RTC requires prefix <= queue threshold < chunk size")
    root = Path(cfg["checkpoint"]).resolve()
    if hardware and (not root.name.isdecimal() or any(p.startswith(".") for p in root.parts)):
        raise ValueError("Hardware execution requires a published numeric checkpoint directory, not staging")
    train = read_json(root / "train_config.json")
    p = train["policy"]
    expected = {
        "type": "openpi_pi05",
        "use_relative_actions": True,
        "relative_exclude_joints": ["gripper"],
        "max_action_dim": 32,
        "chunk_size": cfg["chunk_size"],
        "tokenizer_max_length": 200,
        "dtype": "bfloat16",
        "image_augmentation": False,
        "use_peft": False,
    }
    for key, value in expected.items():
        if p.get(key) != value:
            raise ValueError(f"Unsupported training setting {key}={p.get(key)!r}; expected {value!r}")
    if cfg["rtc"] and not 0 < cfg["rtc_prefix_steps"] <= p.get("rtc_training_max_delay", 0):
        raise ValueError("RTC prefix exceeds the trained delay capacity")
    if cfg["action_names"] != ACTION_NAMES or cfg["camera_names"] != list(CAMERA_NAMES):
        raise ValueError("Deployment channel order differs from the dual-arm training adapter")
    stats = read_json(root / "dataset_stats.json")
    for name in ("action", "observation.state"):
        for quantile in ("q01", "q99"):
            values = stats[name][quantile]
            if len(values) != 16 or any(
                not isinstance(v, (float, int)) or not math.isfinite(v) for v in values
            ):
                raise ValueError(f"{name}.{quantile} must contain 16 finite values")
        if any(b < a for a, b in zip(stats[name]["q01"], stats[name]["q99"], strict=True)):
            raise ValueError(f"{name} has inverted quantiles")
    for name in CAMERA_NAMES:
        if f"observation.images.{name}" not in stats:
            raise ValueError(f"Missing training camera statistics: {name}")
    tokenizer = Path(cfg["tokenizer"])
    for filename in ("tokenizer_config.json", "tokenizer.json"):
        if not (tokenizer / filename).is_file():
            raise FileNotFoundError(tokenizer / filename)
    header = tensor_header(root / "model.safetensors")
    for name, shape in {"action_in_proj.weight": [1024, 32], "action_out_proj.weight": [32, 1024]}.items():
        if name not in header or header[name]["shape"] != shape:
            raise ValueError(f"Not the expected official OpenPI PyTorch checkpoint: {name}")
    return {"train": train, "stats": stats, "tensors": len(header)}
