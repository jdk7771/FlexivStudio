#!/usr/bin/env python
"""Replay one recorded scene on the two follower arms, without cameras or a policy."""

from __future__ import annotations

import argparse
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from deploy_dualarm import align_start_pose, require_compatible_rdk
from flexiv_deploy_contract import ACTION_NAMES, read_json, validate_settings
from flexiv_deploy_hardware import FlexivDualArm

LOGGER = logging.getLogger(__name__)


@dataclass
class Recording:
    scene: str
    source: Path
    timestamps_s: np.ndarray
    targets: np.ndarray
    source_start_ms: float


def load_manifest(cfg: dict) -> dict:
    manifest = read_json(Path(cfg["start_pose"]["manifest"]).expanduser())
    if manifest["dimension_names"] != ACTION_NAMES:
        raise ValueError("Recording dimension order differs from the follower configuration")
    if manifest["units"] != {"joint": "radians", "gripper": "meters"}:
        raise ValueError("Replay requires joint radians and gripper meters")
    for side in ("left", "right"):
        serial = manifest["arms"][side]
        if not serial or not cfg["hardware"]["arms"][side]["serial"].endswith(serial):
            raise ValueError(f"Recorded {side} follower serial differs from the deployment configuration")
    gap = float(manifest["max_lowdim_gap_ms"])
    if not math.isfinite(gap) or gap <= 0:
        raise ValueError("max_lowdim_gap_ms must be finite and positive")
    return manifest


def available_scenes(manifest: dict) -> list[tuple[str, Path, float]]:
    root = Path(manifest["source"]).expanduser()
    choices = []
    for scene in sorted({entry["scene"] for entry in manifest["scenes"]}):
        if Path(scene).name != scene or not scene.startswith("scene_"):
            raise ValueError(f"Invalid recording scene name: {scene!r}")
        source = root / scene / "lowdim/lowdim.h5"
        if not source.is_file():
            LOGGER.warning("Skipping missing recording: %s", source)
            continue
        with h5py.File(source, "r") as lowdim:
            timestamps = lowdim["timestamp"]
            if timestamps.ndim != 1 or len(timestamps) < 2:
                raise ValueError(f"Invalid timestamps in {source}")
            duration = (float(timestamps[-1]) - float(timestamps[0])) / 1000.0
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError(f"Invalid duration in {source}")
        choices.append((scene, source, duration))
    if not choices:
        raise ValueError(f"No recorded scenes found under {root}")
    return choices


def choose_scene(choices: list[tuple[str, Path, float]]) -> str | None:
    print("\n选择要回放的示教数据（仅左右从臂和夹爪运动）：", flush=True)
    for index, (scene, _, duration) in enumerate(choices, 1):
        print(f"  {index:2d}. {scene}  原始时长 {duration:.2f} 秒", flush=True)
    while True:
        try:
            answer = input("输入编号或 scene 名称（q 退出）：").strip()
        except EOFError:
            return None
        if answer.lower() in {"q", "quit"}:
            return None
        if answer.isdecimal() and 1 <= int(answer) <= len(choices):
            return choices[int(answer) - 1][0]
        if any(scene == answer for scene, _, _ in choices):
            return answer
        print("请输入列表中的编号或 scene 名称，或输入 q 退出。", flush=True)


def load_recording(scene: str, source: Path, manifest: dict, cfg: dict) -> Recording:
    """Resample measured follower joints and held gripper commands onto the control clock."""
    with h5py.File(source, "r") as lowdim:
        timestamps = np.asarray(lowdim["timestamp"], dtype=np.float64)
        if (
            timestamps.ndim != 1
            or len(timestamps) < 2
            or not np.isfinite(timestamps).all()
            or np.any(np.diff(timestamps) < 0)
        ):
            raise ValueError(f"Invalid/non-monotonic recording timestamps in {source}")
        # Millisecond timestamp duplicates use the last sample, matching the converter.
        keep = np.r_[np.diff(timestamps) != 0, True]
        unique = timestamps[keep]
        if len(unique) < 2 or np.max(np.diff(unique)) > manifest["max_lowdim_gap_ms"]:
            raise ValueError("Recording has insufficient samples or a lowdim gap; replay refused")
        elapsed = (unique - unique[0]) / 1000.0
        grid = np.arange(0.0, elapsed[-1], 1.0 / cfg["fps"])
        grid = np.r_[grid, elapsed[-1]]
        targets = np.empty((len(grid), 16), dtype=np.float64)
        previous = np.maximum(np.searchsorted(elapsed, grid, side="right") - 1, 0)
        hw = cfg["hardware"]
        for side, offset in (("left", 0), ("right", 8)):
            serial = manifest["arms"][side]
            for prefix, width in (("joint_position_rad", 7), ("ee_command", 1)):
                key = f"{prefix}_{serial}"
                values = np.asarray(lowdim[key], dtype=np.float64)
                if values.shape != (len(timestamps), width) or not np.isfinite(values).all():
                    raise ValueError(f"Invalid recording signal {key}")
                # Permit float32 rounding at the physical bounds, but reject bad units/data.
                if prefix == "ee_command" and (
                    np.any(values < hw["gripper_min_m"] - 1e-6) or np.any(values > hw["gripper_max_m"] + 1e-6)
                ):
                    raise ValueError(f"Recorded {key} is outside the gripper range in meters")
                values = values[keep]
                if prefix == "joint_position_rad":
                    for column in range(7):
                        targets[:, offset + column] = np.interp(grid, elapsed, values[:, column])
                else:
                    targets[:, offset + 7] = np.clip(
                        values[previous, 0], hw["gripper_min_m"], hw["gripper_max_m"]
                    )
    return Recording(scene, source, grid, targets, float(unique[0]))


def validate_speed(speed: float) -> None:
    if not math.isfinite(speed) or not 0 < speed <= 1:
        raise ValueError("--speed must be greater than 0 and at most 1 (1 = original speed)")


def wait_for_start() -> bool:
    while True:
        try:
            answer = input("已到所选数据起点。按 Enter 开始回放，输入 q 退出：").strip().lower()
        except EOFError:
            return False
        if answer in {"q", "quit"}:
            return False
        if not answer:
            return True
        print("按 Enter 开始，或输入 q 退出。", flush=True)


def replay_targets(robot: Any, recording: Recording, cfg: dict, speed: float) -> None:
    validate_speed(speed)
    started = time.monotonic()
    last_sent = None
    next_report = 0.0
    for index, target in enumerate(recording.targets):
        if last_sent is not None:
            # Preserve every target without a burst of catch-up commands after a slow SDK call.
            interval = (recording.timestamps_s[index] - recording.timestamps_s[index - 1]) / speed
            remaining = last_sent + interval - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
        last_sent = time.monotonic()
        robot.send_action(target)
        elapsed = last_sent - started
        if elapsed >= next_report:
            LOGGER.info(
                "%s: frame %d/%d, source %.2f/%.2f s, elapsed %.2f s",
                recording.scene,
                index + 1,
                len(recording.targets),
                recording.timestamps_s[index],
                recording.timestamps_s[-1],
                elapsed,
            )
            next_report = elapsed + 5.0
    robot.wait_for_target(recording.targets[-1], cfg["action_timeout_s"], operation="Replay final pose")
    LOGGER.info("Replay complete: %s (%.2f s elapsed)", recording.scene, time.monotonic() - started)


def execute_replay(cfg: dict, recording: Recording, speed: float) -> None:
    validate_speed(speed)
    require_compatible_rdk()
    robot = FlexivDualArm(cfg["hardware"], start_pose=cfg["start_pose"])
    source = {
        "scene": recording.scene,
        "timestamp_ms": recording.source_start_ms,
        "source": str(recording.source),
    }
    try:
        robot.connect(with_cameras=False)
        # Reject out-of-range joints anywhere in the selected trajectory before alignment.
        for side, offset in (("left", 0), ("right", 8)):
            info = robot.robots[side].info()
            values = recording.targets[:, offset : offset + 7]
            if np.any(values < np.asarray(info.q_min)) or np.any(values > np.asarray(info.q_max)):
                raise ValueError(f"Recorded {side} joints exceed follower joint limits")
        align_start_pose(robot, (recording.targets[0], source))
        robot.hold()
        if not wait_for_start():
            LOGGER.info("Replay cancelled at the recording's first pose")
            return
        # Recheck the actual pose after the potentially long operator wait.
        robot.check_initial_pose(recording.targets[0])
        replay_targets(robot, recording, cfg, speed)
    finally:
        robot.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("deploy_config.json"))
    parser.add_argument("--scene", help="Recorded scene name, e.g. scene_0001; otherwise show a menu")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed: 1 = original, 0.5 = half")
    parser.add_argument("--list", action="store_true", help="List recordings without connecting hardware")
    parser.add_argument(
        "--check", action="store_true", help="Validate the recording without connecting hardware"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    validate_speed(args.speed)
    cfg = read_json(args.config)
    validate_settings(cfg, rtc=False)
    manifest = load_manifest(cfg)
    choices = available_scenes(manifest)
    if args.list:
        for index, (listed_scene, _, duration) in enumerate(choices, 1):
            print(f"{index:2d}. {listed_scene}  {duration:.2f} s")
        return
    scene = args.scene if args.scene is not None else choose_scene(choices)
    if scene is None:
        return
    selected = next((item for item in choices if item[0] == scene), None)
    if selected is None:
        parser.error(f"Scene {scene!r} not found; use --list to see available recordings")
    recording = load_recording(scene, selected[1], manifest, cfg)
    LOGGER.info(
        "Selected %s: %d targets at %d Hz, source %.2f s, speed %.2fx, nominal playback %.2f s",
        scene,
        len(recording.targets),
        cfg["fps"],
        recording.timestamps_s[-1],
        args.speed,
        recording.timestamps_s[-1] / args.speed,
    )
    LOGGER.info("Only configured left/right followers and grippers; no leader arms, cameras or model")
    if args.check:
        LOGGER.info("Recording file check passed. No hardware connected; robot joint limits not checked.")
        return
    execute_replay(cfg, recording, args.speed)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        LOGGER.info("Stopped by operator")
