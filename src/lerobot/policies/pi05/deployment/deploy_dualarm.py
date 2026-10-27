#!/usr/bin/env python
"""Local PI05 trained-RTC deployment. Default: interactive hardware execution.

--check only checks local files. --predict-npz performs one model prediction
from an offline observation and never imports either hardware SDK.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from flexiv_deploy_contract import ACTION_NAMES, CAMERA_NAMES, inspect_checkpoint, read_json

LOGGER = logging.getLogger(__name__)
START_POSE_SETTLE_S = 2.0


def require_compatible_rdk() -> None:
    import flexivrdk

    if flexivrdk.__version__ != "1.9.2":
        raise RuntimeError(
            f"Robot software v3.11.1 requires Flexiv RDK 1.9.2; installed {flexivrdk.__version__}"
        )


def available_checkpoints(models_root: Path) -> list[tuple[str, int, Path, str]]:
    choices = []
    for path in models_root.glob("[0-9]*/pretrained_model"):
        step = path.parent.name
        if not step.isdigit():
            continue
        kind = "LoRA" if (path / "adapter_model.safetensors").is_file() else "full"
        if (path / "config.json").is_file() and (
            (path / "adapter_model.safetensors").is_file() or (path / "model.safetensors").is_file()
        ):
            choices.append((models_root.name, int(step), path, kind))
    return sorted(choices, key=lambda item: -item[1])


def available_checkpoint_folders(models_root: Path) -> list[Path]:
    """List experiment folders containing local weights, excluding download staging folders."""
    return sorted(
        (
            path
            for path in models_root.glob("*")
            if path.is_dir() and not path.name.startswith(".") and available_checkpoints(path)
        ),
        key=lambda path: path.name.casefold(),
    )


def choose_checkpoint(models_root: Path) -> Path | None:
    folders = available_checkpoint_folders(models_root)
    if folders:
        print(f"\n选择权重文件夹（{models_root}）：", flush=True)
        for index, folder in enumerate(folders, 1):
            print(f"  {index}. {folder.name}", flush=True)
        while True:
            try:
                answer = input("输入文件夹编号（q 退出）：").strip().lower()
            except EOFError as error:
                raise SystemExit("需要交互输入；也可用 --checkpoint 直接指定权重") from error
            if answer in {"q", "quit"}:
                return None
            if answer.isdecimal() and 1 <= int(answer) <= len(folders):
                models_root = folders[int(answer) - 1]
                break
            print(f"请输入 1-{len(folders)}，或输入 q 退出。", flush=True)
    # Retain support for custom configs whose checkpoint_root is one experiment folder.
    choices = available_checkpoints(models_root)
    if not choices:
        raise FileNotFoundError(f"No local checkpoints found under {models_root}")
    print(f"\n选择 {models_root.name} 的 checkpoint：", flush=True)
    for index, (experiment, step, path, kind) in enumerate(choices, 1):
        print(f"  {index}. {experiment}  step {step:,}  {kind}  {path}", flush=True)
    while True:
        try:
            answer = input("输入编号（q 退出）：").strip().lower()
        except EOFError as error:
            raise SystemExit("需要交互输入；也可用 --checkpoint 直接指定权重") from error
        if answer in {"q", "quit"}:
            return None
        if answer.isdecimal() and 1 <= int(answer) <= len(choices):
            return choices[int(answer) - 1][2]
        print(f"请输入 1-{len(choices)}，或输入 q 退出。", flush=True)


def choose_run_mode() -> str | None:
    print("\n选择运行方式：\n  1. 单步：每按一次 Enter 下发一个新动作\n  2. 连续：按 15 Hz 运行", flush=True)
    while True:
        try:
            answer = input("输入 1/2（q 退出）：").strip().lower()
        except EOFError as error:
            raise SystemExit("需要交互输入；也可用 --run-mode step/continuous") from error
        if answer in {"q", "quit"}:
            return None
        if answer in {"1", "2"}:
            return "step" if answer == "1" else "continuous"
        print("请输入 1、2 或 q。", flush=True)


def wait_for_continuous_start() -> bool:
    """Keep the followers at the recorded home pose until Enter starts the policy."""
    while True:
        try:
            answer = input("已回到 home（采集首帧）。按 Enter 开始连续运行，输入 q 退出：").strip().lower()
        except EOFError:
            LOGGER.info("No start key received; deployment cancelled at home")
            return False
        if answer in {"q", "quit"}:
            return False
        if not answer:
            return True
        print("按 Enter 开始连续运行，或输入 q 退出。", flush=True)


def read_start_pose(cfg: dict) -> tuple[Any, dict]:
    """Recreate the first retained observation.state from the source HDF5 at each launch."""
    import h5py
    import numpy as np

    start = cfg["start_pose"]
    manifest = read_json(Path(start["manifest"]))
    if manifest["dimension_names"] != ACTION_NAMES or manifest["fps"] != cfg["fps"]:
        raise ValueError("Collection manifest does not match the deployed state order or FPS")
    if manifest["task"] != cfg["task"]:
        raise ValueError("Collection manifest task differs from the selected policy task")
    for side in ("left", "right"):
        if not cfg["hardware"]["arms"][side]["serial"].endswith(manifest["arms"][side]):
            raise ValueError(f"Collection {side} arm differs from the deployed follower")
    scene = next((item for item in manifest["scenes"] if item["scene"] == start["scene"]), None)
    if scene is None or not scene["segments"]:
        raise ValueError(f"No retained first frame for {start['scene']}")
    timestamp_ms = float(scene["segments"][0]["source_start_ms"])
    source = Path(manifest["source"]) / start["scene"] / "lowdim/lowdim.h5"
    with h5py.File(source, "r") as lowdim:
        timestamps = np.asarray(lowdim["timestamp"], dtype=np.float64).reshape(-1)
        if len(timestamps) < 2 or not np.isfinite(timestamps).all() or (np.diff(timestamps) < 0).any():
            raise ValueError("Invalid collection timestamps")
        keep = np.r_[np.diff(timestamps) != 0, True]
        timestamps = timestamps[keep]
        if not timestamps[0] <= timestamp_ms <= timestamps[-1]:
            raise ValueError("The first retained frame is outside the collection timestamps")
        right = int(np.searchsorted(timestamps, timestamp_ms))
        if (
            right < len(timestamps)
            and timestamps[right] != timestamp_ms
            and (timestamps[right] - timestamps[right - 1] > manifest["max_lowdim_gap_ms"])
        ):
            raise ValueError("The collection's first frame crosses a lowdim sampling gap")
        parts: list[float] = []
        for side in ("left", "right"):
            serial = manifest["arms"][side]
            for prefix, width in (("joint_position_rad", 7), ("ee_state", 1)):
                values = np.asarray(lowdim[f"{prefix}_{serial}"], dtype=np.float32)
                if values.shape != (len(keep), width) or not np.isfinite(values).all():
                    raise ValueError(f"Invalid collection signal {prefix}_{serial}")
                parts.extend(
                    np.interp(timestamp_ms, timestamps, values[keep, column]) for column in range(width)
                )
    state = np.asarray(parts, dtype=np.float32)
    if state.shape != (16,) or not np.isfinite(state).all():
        raise ValueError("Invalid first-frame joint/gripper state")
    return state, {"scene": start["scene"], "timestamp_ms": timestamp_ms, "source": str(source)}


def load_policy(cfg: dict, description: dict) -> tuple[Any, Any, Any]:
    # Stay entirely offline, including accidental stale server paths in saved configs.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    from safetensors.torch import load_file, load_model
    from transformers.initialization import no_init_weights

    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.policies.rtc.configuration_rtc import RTCConfig

    if not cfg["device"].startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("This local deployment requires a working CUDA GPU/driver")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected CUDA GPU must support BF16")
    policy_cfg = PI05Config.from_pretrained(cfg["checkpoint"], local_files_only=True)
    policy_cfg.device, policy_cfg.dtype = cfg["device"], torch.bfloat16
    policy_cfg.action_feature_names = ACTION_NAMES.copy()
    policy_cfg.text_tokenizer_name = cfg["tokenizer"]
    policy_cfg.gradient_checkpointing = False
    policy_cfg.compile_model = False
    policy_cfg.num_inference_steps = cfg["num_inference_steps"]
    policy_cfg.rtc_config = RTCConfig(
        enabled=True, mode="trained", execution_horizon=cfg["rtc_execution_horizon"]
    )
    # The upstream PI05 loader catches loading failures and can return random weights.
    # Load explicitly with strict=True so this deployment always fails on bad weights.
    # Every persistent parameter is loaded strictly below; initializing random weights first is expensive.
    with no_init_weights():
        policy = PI05Policy(policy_cfg)
    weight_root = cfg["base_model"] if description["kind"] == "LoRA" else cfg["checkpoint"]
    LOGGER.info("Loading %s model weights from %s", description["kind"], weight_root)
    if description["kind"] == "LoRA":
        weights = load_file(str(Path(weight_root) / "model.safetensors"), device="cpu")
        weights = policy._convert_openpi_state_dict(weights)
        policy.load_state_dict(weights, strict=True)
        del weights
    else:
        # Native saves can deduplicate tied embeddings; safetensors restores these aliases.
        load_model(policy, str(Path(weight_root) / "model.safetensors"), strict=True, device="cpu")
    if description["kind"] == "LoRA":
        from flexiv_deploy_contract import tensor_header
        from peft import PeftConfig, PeftModel, get_peft_model_state_dict

        adapter_cfg = PeftConfig.from_pretrained(cfg["checkpoint"], local_files_only=True)
        # This path is metadata only here; the base was loaded explicitly from the local override.
        adapter_cfg.base_model_name_or_path = cfg["base_model"]
        policy = PeftModel.from_pretrained(
            policy, cfg["checkpoint"], config=adapter_cfg, is_trainable=False, local_files_only=True
        )
        expected = tensor_header(Path(cfg["checkpoint"]) / "adapter_model.safetensors")
        actual = get_peft_model_state_dict(policy)
        if set(actual) != set(expected) or any(list(actual[k].shape) != expected[k]["shape"] for k in actual):
            raise ValueError("Saved LoRA tensors do not exactly match this adapter architecture")
    policy.to(cfg["device"]).eval()
    expected_device = torch.device(cfg["device"])
    actual_devices = {parameter.device for parameter in policy.parameters()}
    if actual_devices != {expected_device}:
        raise RuntimeError(f"Policy parameters are not entirely on {expected_device}: {actual_devices}")
    LOGGER.info(
        "Policy parameters loaded on %s (%s)", expected_device, torch.cuda.get_device_name(expected_device)
    )
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=cfg["checkpoint"],
        preprocessor_overrides={
            "device_processor": {"device": cfg["device"]},
            "tokenizer_processor": {"tokenizer_name": cfg["tokenizer"]},
            "relative_actions_processor": {"action_names": ACTION_NAMES.copy()},
        },
    )
    # RTCInferenceEngine owns both pipelines on one worker and re-anchors each prefix.
    policy.reset()
    return policy, preprocessor, postprocessor


def dataset_features() -> dict:
    return {
        "observation.state": {"dtype": "float32", "shape": (16,), "names": ACTION_NAMES.copy()},
        "action": {"dtype": "float32", "shape": (16,), "names": ACTION_NAMES.copy()},
        **{f"observation.images.{name}": {"dtype": "image", "shape": (224, 224, 3)} for name in CAMERA_NAMES},
    }


def predict_npz(
    cfg: dict, policy: Any, preprocessor: Any, postprocessor: Any, path: Path, output: Path
) -> None:
    """One offline, prefix-free chunk, in physical absolute units; not a closed-loop RTC test."""
    import numpy as np

    from lerobot.inference.prediction import chunk_inference_context, predict_chunk
    from lerobot.policies import ExecutionMode, prepare_observation_for_inference

    with np.load(path, allow_pickle=False) as saved:
        state = saved["observation.state"].copy()
        if state.shape != (16,) or not np.isfinite(state).all():
            raise ValueError("Offline observation.state must be a finite (16,) array")
        observation = {"observation.state": state}
        for name in CAMERA_NAMES:
            key = f"observation.images.{name}"
            image = saved[key].copy()
            if image.shape != (224, 224, 3) or image.dtype != np.uint8:
                raise ValueError(f"{key} must be letterboxed RGB uint8, shape (224,224,3)")
            observation[key] = image
    with chunk_inference_context(ExecutionMode.RTC_TRAINED):
        batch = prepare_observation_for_inference(observation, cfg["device"], cfg["task"], "flexiv_dual_arm")
        batch["task"] = [cfg["task"]]
        prediction = predict_chunk(
            policy,
            postprocessor,
            preprocessor(batch),
            spec=policy.chunk_inference_spec(),
            mode=ExecutionMode.RTC_TRAINED,
            canonical_action_dim=16,
            model_action_dim=None,
            device=cfg["device"],
            rtc_horizon=cfg["rtc_execution_horizon"],
        )
    # Save only the real 16 channels, not the model's padded 32 channels.
    with output.open("xb") as stream:
        np.save(stream, prediction.canonical_actions[:, :16].cpu().numpy(), allow_pickle=False)
    LOGGER.info("Saved absolute joint/gripper targets to %s; no hardware was connected", output)


def dispatch_single_step(engine: Any, robot: Any, cfg: dict) -> tuple[dict, Any, float]:
    """Predict from a fresh observation and dispatch exactly one action."""
    engine.reset()
    engine.resume()
    deadline = time.monotonic() + cfg["action_timeout_s"]
    try:
        while time.monotonic() < deadline:
            if engine.failed:
                raise RuntimeError(engine.failure_traceback)
            observation = robot.get_observation()
            engine.notify_observation(observation)
            if engine.dispatch_allowed():
                action = engine.get_action(None)
                if action is not None and engine.dispatch_allowed():
                    applied = robot.send_action(action.detach().float().cpu().numpy()[:16])
                    return observation, applied, robot.observation_time
            else:
                engine.acknowledge_hold()
            robot.hold()
            time.sleep(1 / cfg["fps"])
        raise TimeoutError("No fresh RTC action was ready for this step")
    finally:
        engine.pause()


def align_start_pose(robot: Any, start_pose: tuple[Any, dict], *, close_grippers: bool = False) -> None:
    import numpy as np

    initial, source = start_pose
    measured, distance = robot.check_initial_pose(initial)
    LOGGER.info(
        "Collection first frame: %s at %.1f ms; max joint difference %.3f rad",
        source["scene"],
        source["timestamp_ms"],
        distance,
    )
    LOGGER.info("Target grippers: left %.4f m, right %.4f m", initial[7], initial[15])
    LOGGER.info(
        "Left/right joint difference: %.3f / %.3f rad",
        np.max(np.abs(initial[:7] - measured[:7])),
        np.max(np.abs(initial[8:15] - measured[8:15])),
    )
    if close_grippers:
        closing = measured.copy()
        closing[[7, 15]] = initial[[7, 15]]
        robot.send_action(closing, force_gripper_command=True)
        LOGGER.info("Both grippers commanded closed before arm alignment")
    aligned = robot.move_to_initial_pose(initial)
    LOGGER.info(
        "Initial pose reached; max joint error %.3f rad",
        np.max(np.abs(initial[np.r_[0:7, 8:15]] - aligned[np.r_[0:7, 8:15]])),
    )


def execute(
    cfg: dict,
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    run_mode: str,
    start_pose: tuple[Any, dict],
) -> None:
    import numpy as np
    from flexiv_deploy_hardware import FlexivDualArm

    from lerobot.inference import RTCInferenceEngine
    from lerobot.policies.rtc.configuration_rtc import RTCConfig

    robot = FlexivDualArm(cfg["hardware"], start_pose=cfg["start_pose"])
    engine = None
    initial, source = start_pose
    telemetry = Path(cfg["telemetry"])
    telemetry.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Engine construction deep-copies processors/tokenizers. Finish that work
        # before camera reader threads and robot control are active.
        engine_init_start = time.monotonic()
        engine = RTCInferenceEngine(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            robot_wrapper=robot,
            dataset_features=dataset_features(),
            task=cfg["task"],
            fps=cfg["fps"],
            device=cfg["device"],
            rtc_config=RTCConfig(
                enabled=True, mode="trained", execution_horizon=cfg["rtc_execution_horizon"]
            ),
            rtc_queue_threshold=cfg["rtc_queue_threshold"],
            max_observation_age_s=cfg["max_observation_age_s"],
            action_timeout_s=cfg["action_timeout_s"],
            action_starvation_grace_s=0.0,
        )
        LOGGER.info(
            "RTC engine prepared before device connection in %.2f s", time.monotonic() - engine_init_start
        )
        robot.connect()
        align_start_pose(robot, start_pose)
        LOGGER.info("Initial pose reached; waiting %.1f s to settle before inference", START_POSE_SETTLE_S)
        time.sleep(START_POSE_SETTLE_S)
        if run_mode == "continuous":
            robot.hold()
            if not wait_for_continuous_start():
                LOGGER.info("Deployment cancelled before policy execution")
                return
        engine.start()
        # Require NEW frames after setup and any operator wait before policy execution.
        robot.wait_for_fresh_observation(timeout_s=cfg["hardware"]["connect_timeout_s"])
        if run_mode == "continuous":
            engine.resume()
        started = deadline = time.monotonic()
        last_report = started
        with telemetry.open("a", encoding="utf-8", buffering=1) as log:
            if run_mode == "step":
                step = 0
                while True:
                    answer = input(f"单步 {step + 1}：按 Enter 执行一个动作，输入 q 退出：").strip().lower()
                    if answer in {"q", "quit"}:
                        break
                    if answer:
                        LOGGER.info("Only Enter dispatches one step; no action sent")
                        continue
                    obs, target, capture_time = dispatch_single_step(engine, robot, cfg)
                    camera_age_s = time.monotonic() - capture_time
                    robot.wait_for_target(target, cfg["action_timeout_s"])
                    step += 1
                    log.write(
                        json.dumps(
                            {
                                "wall_time": time.time(),
                                "mode": "step",
                                "step": step,
                                "start_scene": source["scene"],
                                "state": [float(obs[name]) for name in ACTION_NAMES],
                                "applied_absolute_action": np.asarray(target).tolist(),
                                "camera_age_s": camera_age_s,
                            }
                        )
                        + "\n"
                    )
                    LOGGER.info("Step %d reached target; RTC queue cleared", step)
                return
            while time.monotonic() - started < cfg["duration_s"]:
                tick_start = time.monotonic()
                if engine.failed:
                    raise RuntimeError(engine.failure_traceback)
                obs = robot.get_observation()
                capture_done = time.monotonic()
                engine.notify_observation(obs)
                target = None
                if engine.dispatch_allowed():
                    action = engine.get_action(None)
                    # get_action() itself may revoke dispatch when the buffer runs out.
                    if action is not None and engine.dispatch_allowed():
                        target = robot.send_action(action.detach().float().cpu().numpy()[:16])
                    else:
                        robot.hold()
                else:
                    robot.hold()
                    engine.acknowledge_hold()
                if engine.failed:
                    raise RuntimeError(engine.failure_traceback)
                ended = time.monotonic()
                log.write(
                    json.dumps(
                        {
                            "wall_time": time.time(),
                            "mode": "continuous",
                            "start_scene": source["scene"],
                            "elapsed_s": ended - started,
                            "state": [float(obs[name]) for name in ACTION_NAMES],
                            "applied_absolute_action": None
                            if target is None
                            else np.asarray(target).tolist(),
                            "camera_age_s": capture_done - robot.observation_time,
                            "camera_ages_s": robot.camera_ages_s.copy(),
                            "observation_ms": (capture_done - tick_start) * 1000,
                            "command_ms": (ended - capture_done) * 1000,
                            "queue_size": engine.action_queue.qsize(),
                        }
                    )
                    + "\n"
                )
                if ended - last_report >= 5:
                    LOGGER.info(
                        "elapsed=%.1fs queue=%d tick=%.1fms",
                        ended - started,
                        engine.action_queue.qsize(),
                        (ended - tick_start) * 1000,
                    )
                    last_report = ended
                deadline += 1 / cfg["fps"]
                remaining = deadline - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)
                else:
                    # Never accelerate queued actions to catch up after a late control tick.
                    deadline = time.monotonic() + 1 / cfg["fps"]
                    LOGGER.warning("Control tick missed 15 Hz budget by %.1f ms", -remaining * 1000)
                    time.sleep(1 / cfg["fps"])
    finally:
        # Stop robot motion before joining a potentially busy GPU worker.
        try:
            if engine is not None:
                engine.pause()
        finally:
            robot.close()
            if engine is not None:
                engine.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("deploy_config.json"))
    parser.add_argument("--checkpoint", type=Path, help="Skip the interactive checkpoint menu")
    parser.add_argument("--base-model", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument(
        "--run-mode", choices=("step", "continuous"), help="Skip the interactive run-mode menu"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Check local files only")
    mode.add_argument("--predict-npz", type=Path, help="Predict from an offline NPZ; never connect hardware")
    parser.add_argument("--output", type=Path, help="New .npy path required with --predict-npz")
    args = parser.parse_args()
    if args.predict_npz is not None and args.output is None:
        parser.error("--predict-npz requires --output")
    execute_mode = not args.check and args.predict_npz is None
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = read_json(args.config)
    if execute_mode and args.checkpoint is None:
        checkpoint_root = cfg.get("checkpoint_root", str(Path(cfg["checkpoint"]).parent))
        selected = choose_checkpoint(Path(checkpoint_root).expanduser())
        if selected is None:
            LOGGER.info("Deployment cancelled before loading the model or connecting devices")
            return
        args.checkpoint = selected
    if execute_mode and args.run_mode is None:
        args.run_mode = choose_run_mode()
        if args.run_mode is None:
            LOGGER.info("Deployment cancelled before loading the model or connecting devices")
            return
    if execute_mode:
        require_compatible_rdk()
    for name in ("checkpoint", "base_model", "tokenizer"):
        override = getattr(args, name)
        cfg[name] = str((override or Path(cfg[name])).expanduser().resolve())
    if args.tokenizer is None and (Path(cfg["checkpoint"]) / "tokenizer").is_dir():
        cfg["tokenizer"] = str(Path(cfg["checkpoint"]) / "tokenizer")
    description = inspect_checkpoint(cfg)
    LOGGER.info("Checkpoint files valid: %s (%s)", cfg["checkpoint"], description["kind"])
    if args.check:
        LOGGER.info("File checks complete. No model loaded, no devices connected.")
        return
    start_pose = read_start_pose(cfg) if execute_mode else None
    policy, preprocessor, postprocessor = load_policy(cfg, description)
    if args.predict_npz is not None:
        predict_npz(cfg, policy, preprocessor, postprocessor, args.predict_npz, args.output)
    else:
        if start_pose is None:
            raise ValueError("Hardware execution requires the recorded starting pose")
        execute(cfg, policy, preprocessor, postprocessor, args.run_mode, start_pose)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        LOGGER.info("Stopped by operator")
