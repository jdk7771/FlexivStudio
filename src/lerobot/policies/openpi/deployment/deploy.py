#!/usr/bin/env python
"""Independent OpenPI dual-arm deployment. Check/predict modes never connect hardware."""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import shared  # noqa: F401

# isort: split

import numpy as np
from deploy_dualarm_no_rtc import read_start_pose, require_compatible_rdk, wait_for_continuous_start
from deployment_recording import DeploymentLog, add_recording_arguments, configure_recording
from flexiv_deploy_contract import ACTION_NAMES, CAMERA_NAMES, read_json

from contract import choose_checkpoint, inspect, select
from engine import ChunkEngine

LOGGER = logging.getLogger(__name__)


def home(robot: Any, cfg: dict, start_pose: tuple) -> None:
    """Use exactly goHome's float64 targets, raised path and verified gripper closure."""
    from go_home_path import read_home_tcp, return_to_home

    recorded, source = start_pose
    tcp = read_home_tcp(cfg, source)
    return_to_home(robot, cfg, recorded, tcp, cfg["home_lift_m"], skip_if_close=True)
    robot.hold()


def read_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        state = data["observation.state"]
        if state.shape != (16,) or not np.isfinite(state).all():
            raise ValueError("observation.state must contain 16 finite values")
        result = dict(zip(ACTION_NAMES, state, strict=True))
        for name in CAMERA_NAMES:
            result[name] = data[f"observation.images.{name}"].copy()
    return result


def execute(cfg: dict, policy: Any, run_mode: str, start_pose: tuple) -> None:
    from flexiv_deploy_hardware import FlexivDualArm
    from high_rate_state_recording import StateRecorder

    root = Path(cfg["log_root"]) / (datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8])
    root.mkdir(parents=True, exist_ok=False)
    cfg = {**cfg, "telemetry": str(root / "telemetry.jsonl"), "run_mode": run_mode}
    # Persist the selected checkpoint for EVERY run, even without image recording.
    (root / "metadata.json").write_text(json.dumps(cfg, indent=2) + "\n")
    robot = FlexivDualArm(cfg["hardware"], start_pose=cfg["start_pose"])
    log = engine = recorder = None
    try:
        log = DeploymentLog(cfg, run_mode, rtc=cfg["rtc"])
        robot.connect()
        if cfg["state_hz"]:
            recorder = StateRecorder(root / "robot_states", cfg, hz=cfg["state_hz"])
            state_meta_path = root / "robot_states" / "metadata.json"
            state_meta = read_json(state_meta_path)
            state_meta["rtc"] = cfg["rtc"]
            state_meta_path.write_text(json.dumps(state_meta, indent=2) + "\n")
            recorder.start(robot.robots, robot.grippers)
        home(robot, cfg, start_pose)
        time.sleep(2.0)
        if run_mode == "continuous" and not wait_for_continuous_start():
            return
        robot.wait_for_fresh_observation(timeout_s=cfg["hardware"]["connect_timeout_s"])
        if run_mode == "continuous":
            engine = ChunkEngine(policy, cfg)
        start = deadline = time.monotonic()
        step = 0
        while run_mode == "step" or time.monotonic() - start < cfg["duration_s"]:
            if run_mode == "step":
                try:
                    answer = input("按 Enter 推理并执行一个新动作（q 退出）：").strip().lower()
                except EOFError:
                    break
                if answer in {"q", "quit"}:
                    break
                if answer:
                    continue
            obs = robot.get_observation()
            capture = robot.observation_time
            if capture is None:
                raise ValueError("Missing camera observation time")
            capture_ages = robot.camera_ages_s.copy()
            started = time.monotonic()
            if run_mode == "step":
                robot.hold()
                actions = policy.predict(obs)
                robot.get_observation()  # Recheck camera freshness and arm status after inference.
                if (
                    time.monotonic() - started > cfg["action_timeout_s"]
                    or time.monotonic() - capture > cfg["max_observation_age_s"]
                ):
                    raise TimeoutError("Single-step inference expired; no command sent")
                target = robot.send_action(actions[0])
                robot.wait_for_target(target, cfg["action_timeout_s"])
            else:
                if engine is None:
                    raise RuntimeError("Continuous execution requires an inference engine")
                action = engine.tick(obs, capture)
                if time.monotonic() - start >= cfg["duration_s"]:
                    break
                if action is None:
                    robot.hold()
                    target = None
                else:
                    target = robot.send_action(action)
            step += 1
            log.record(
                {
                    "wall_time": time.time(),
                    "elapsed_s": time.monotonic() - start,
                    "mode": run_mode,
                    "step": step,
                    "checkpoint": cfg["checkpoint"],
                    "rtc": cfg["rtc"],
                    "start_scene": start_pose[1]["scene"],
                    "state": [float(obs[n]) for n in ACTION_NAMES],
                    "applied_absolute_action": None if target is None else target.tolist(),
                    "camera_age_s": started - capture,
                    "camera_ages_s": capture_ages,
                    "queue_size": 0 if engine is None else len(engine.queue),
                    "inference_ms": (time.monotonic() - started) * 1000
                    if engine is None
                    else engine.last_inference_ms,
                    "inference_count": step if engine is None else engine.inferences,
                    "prefix_length": 0 if engine is None else engine.last_prefix_length,
                    "rtc_discarded_results": 0 if engine is None else engine.discarded,
                    "rtc_trimmed_actions": 0 if engine is None else engine.last_trimmed,
                    "rtc_turnaround_estimate_ms": 0 if engine is None else engine.turnaround * 1000,
                    "control_ms": (time.monotonic() - started) * 1000,
                },
                obs,
                capture,
            )
            if run_mode == "continuous":
                deadline += 1 / cfg["fps"]
                now = time.monotonic()
                if deadline <= now:
                    # An overrun starts a new period; never catch up by sending the next point immediately.
                    deadline = now + 1 / cfg["fps"]
                time.sleep(max(0.0, deadline - now))
    finally:
        # Stop followers before waiting for inference or disk flushes.
        if recorder is not None:
            recorder.request_stop()
        try:
            robot.close()
        finally:
            try:
                if engine is not None:
                    engine.close()
            finally:
                try:
                    if recorder is not None:
                        recorder.close()
                finally:
                    try:
                        if log is not None:
                            log.close()
                    finally:
                        LOGGER.info("Run metadata and telemetry: %s", root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("deploy_config.json"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--rtc", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--n-action-steps", type=int)
    parser.add_argument("--run-mode", choices=("step", "continuous"))
    parser.add_argument("--duration-s", type=float)
    parser.add_argument(
        "--state-hz", type=int, default=0, help="0 disables state-only high-rate recording; 1..1000 Hz"
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--check", action="store_true")
    modes.add_argument("--predict-npz", type=Path)
    parser.add_argument("--output", type=Path)
    add_recording_arguments(parser)
    args = parser.parse_args()
    if args.predict_npz and args.output is None:
        parser.error("--predict-npz requires --output")
    if not 0 <= args.state_hz <= 1000:
        parser.error("--state-hz must be in [0,1000]")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = read_json(args.config)
    for key in ("rtc", "n_action_steps", "duration_s"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    cfg["state_hz"] = args.state_hz
    hardware = not args.check and args.predict_npz is None
    if hardware and args.checkpoint is None:
        args.checkpoint = choose_checkpoint(Path(cfg["checkpoint_root"]))
        if args.checkpoint is None:
            return
    for key in ("checkpoint", "tokenizer"):
        cfg[key] = str((getattr(args, key) or Path(cfg[key])).expanduser().resolve())
    if hardware and args.run_mode is None:
        mode = select("选择运行方式：", ["单步：每次 Enter 执行一个新动作", "连续：15 Hz"])
        if mode is None:
            return
        args.run_mode = "step" if mode.startswith("单步") else "continuous"
    if args.run_mode == "step":
        cfg["rtc"] = False  # Fresh, prefix-free inference per keypress.
    from go_home_path import validate_lift

    validate_lift(cfg["home_lift_m"])
    configure_recording(cfg, args)
    description = inspect(cfg, hardware=hardware)
    LOGGER.info(
        "Official OpenPI checkpoint: %s; RTC=%s; training delay=%s",
        cfg["checkpoint"],
        cfg["rtc"],
        description["train"]["policy"]["rtc_training_max_delay"],
    )
    if args.check:
        LOGGER.info("Checkpoint/config/tokenizer validated; no model or hardware loaded")
        return
    start_pose = None
    if hardware:
        require_compatible_rdk()
        start_pose = read_start_pose(cfg)
    from policy import OpenPiPolicy

    policy = OpenPiPolicy(cfg, description)
    # Warm CUDA kernels before connecting hardware, using synthetic images only for warmup.
    if args.predict_npz:
        actions = policy.predict(read_npz(args.predict_npz))
        with args.output.open("xb") as stream:
            np.save(stream, actions, allow_pickle=False)
        LOGGER.info("Saved absolute targets %s to %s; no hardware connected", actions.shape, args.output)
        return
    if start_pose is None:
        raise ValueError("Hardware execution requires a recorded home pose")
    warm = dict(zip(ACTION_NAMES, start_pose[0], strict=True))
    warm.update({name: np.zeros((224, 224, 3), dtype=np.uint8) for name in CAMERA_NAMES})
    began = time.monotonic()
    policy.predict(warm)
    LOGGER.info("Offline warmup %.3f s; now connecting followers/cameras", time.monotonic() - began)
    execute(cfg, policy, args.run_mode, start_pose)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        LOGGER.info("Stopped by operator")
