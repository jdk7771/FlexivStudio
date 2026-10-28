"""Offline and mocked tests for the standalone non-RTC deployment."""

from __future__ import annotations

import importlib.util
import io
import sys
import tempfile
import unittest
from collections import deque
from contextlib import redirect_stdout
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch

import numpy as np
from deploy_dualarm_no_rtc import (
    dispatch_single_step,
    execute as execute_deployment,
    main as deploy_main,
    next_policy_action,
)
from flexiv_deploy_contract import ACTION_NAMES, read_json
from flexiv_deploy_hardware import CameraFreshnessError


class TestNonRTCEntry(unittest.TestCase):
    def test_hardware_utilities_accept_non_rtc_config_without_connecting(self) -> None:
        import goHome
        import init_grippers
        import replay_data

        config = Path(__file__).resolve().parents[1] / "configs/deploy_config_no_rtc.json"
        start_pose = (np.zeros(16), {"scene": "scene_0001", "timestamp_ms": 150.0})
        with (
            patch("goHome.read_start_pose", return_value=start_pose),
            patch("goHome.read_home_tcp", return_value=np.zeros((2, 7))),
            patch("replay_data.load_manifest", return_value={}),
            patch("replay_data.available_scenes", return_value=[]),
        ):
            for module, flag in ((goHome, "--check"), (init_grippers, "--check"), (replay_data, "--list")):
                with (
                    self.subTest(entry=module.__name__),
                    patch.object(sys, "argv", [module.__name__, "--config", str(config), flag]),
                    patch.object(module, "require_compatible_rdk") as sdk,
                    patch.object(module, "FlexivDualArm") as hardware,
                ):
                    module.main()
                sdk.assert_not_called()
                hardware.assert_not_called()

    def test_command_line_overrides_configured_execution_steps(self) -> None:
        with (
            patch.object(sys, "argv", ["deploy_dualarm_no_rtc.py", "--n-action-steps", "5", "--check"]),
            patch("deploy_dualarm_no_rtc.inspect_checkpoint", return_value={"kind": "full"}) as inspect,
            patch("deploy_dualarm_no_rtc.load_policy") as load_policy,
        ):
            deploy_main()
        self.assertEqual(inspect.call_args.args[0]["n_action_steps"], 5)
        load_policy.assert_not_called()

    def test_default_check_uses_non_rtc_config_without_loading_or_connecting(self) -> None:
        with (
            patch.object(sys, "argv", ["deploy_dualarm_no_rtc.py", "--check"]),
            patch("deploy_dualarm_no_rtc.inspect_checkpoint", return_value={"kind": "full"}) as inspect,
            patch("deploy_dualarm_no_rtc.require_compatible_rdk") as require_rdk,
            patch("deploy_dualarm_no_rtc.load_policy") as load_policy,
            patch("deploy_dualarm_no_rtc.execute") as execute,
        ):
            deploy_main()
        self.assertEqual(inspect.call_args.kwargs, {"rtc": False})
        cfg = inspect.call_args.args[0]
        self.assertIn("DualArm_FoldTowel_pi05_no_rtc_full/025000", cfg["checkpoint"])
        self.assertTrue(cfg["telemetry"].endswith("DualArm_FoldTowel_no_rtc_full.jsonl"))
        self.assertNotIn("rtc_execution_horizon", cfg)
        require_rdk.assert_not_called()
        load_policy.assert_not_called()
        execute.assert_not_called()

    def test_menu_cancel_exits_before_model_or_hardware_checks(self) -> None:
        with (
            patch.object(sys, "argv", ["deploy_dualarm_no_rtc.py"]),
            patch("deploy_dualarm_no_rtc.choose_checkpoint", return_value=None),
            patch("deploy_dualarm_no_rtc.inspect_checkpoint") as inspect,
            patch("deploy_dualarm_no_rtc.require_compatible_rdk") as require_rdk,
            patch("deploy_dualarm_no_rtc.load_policy") as load_policy,
        ):
            deploy_main()
        inspect.assert_not_called()
        require_rdk.assert_not_called()
        load_policy.assert_not_called()


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch required for non-RTC inference")
class TestNonRTCInference(unittest.TestCase):
    def setUp(self) -> None:
        import torch

        from lerobot.inference import SyncInferenceEngine
        from lerobot.policies.pretrained import PreTrainedPolicy
        from lerobot.processor import (
            AbsoluteActionsProcessorStep,
            RelativeActionsProcessorStep,
            TransitionKey,
            bind_relative_anchor,
            create_transition,
        )

        self.cfg = read_json(Path(__file__).resolve().parents[1] / "configs/deploy_config_no_rtc.json")
        self.clock = [100.0]
        self.enterContext(patch("deploy_dualarm_no_rtc.time.monotonic", side_effect=lambda: self.clock[0]))
        self.observation = dict.fromkeys(ACTION_NAMES, 0.5)
        self.observation.update(
            {name: np.zeros((224, 224, 3), dtype=np.uint8) for name in ("image", "image2", "image3")}
        )
        self.robot = MagicMock()
        self.robot.camera_ages_s = {}

        def capture() -> dict:
            self.robot.observation_time = self.clock[0]
            return self.observation.copy()

        self.robot.get_observation.side_effect = capture
        self.robot.send_action.side_effect = lambda action: action
        self.queue: deque[torch.Tensor] = deque()
        self.predictions = 0
        self.prediction_delay = 0.0

        def select_action(batch: dict) -> torch.Tensor:
            if not self.queue:
                self.predictions += 1
                self.clock[0] += self.prediction_delay
                for offset in (0.1, 0.2, 0.3):
                    action = torch.full((1, 16), offset)
                    action[:, [7, 15]] = 0.04
                    self.queue.append(action)
            return self.queue.popleft()

        self.policy = SimpleNamespace(
            config=SimpleNamespace(use_amp=False),
            count_queued_actions=lambda: len(self.queue),
            select_action=select_action,
            reset=self.queue.clear,
        )
        relative = RelativeActionsProcessorStep(
            enabled=True, exclude_joints=["gripper"], action_names=ACTION_NAMES.copy()
        )
        absolute = AbsoluteActionsProcessorStep(enabled=True, relative_step=relative)

        def preprocess(observation: dict) -> dict:
            relative(create_transition(observation=observation))
            return observation

        pre = MagicMock(steps=[relative], side_effect=preprocess)
        pre.reset.side_effect = relative.reset
        post = MagicMock(
            side_effect=lambda action: absolute(create_transition(action=action))[TransitionKey.ACTION]
        )
        mock_policy = cast(PreTrainedPolicy, self.policy)
        bind_relative_anchor(mock_policy, pre)
        self.engine = SyncInferenceEngine(
            policy=mock_policy,
            preprocessor=pre,
            postprocessor=post,
            dataset_features={},
            ordered_action_keys=ACTION_NAMES.copy(),
            task=self.cfg["task"],
            device="cpu",
            robot_type="flexiv_dual_arm",
        )

    def _next_action(self) -> np.ndarray:
        observation = self.robot.get_observation()
        return next_policy_action(self.engine, self.robot, self.cfg, observation, self.policy).numpy()

    def test_chunk_keeps_original_joint_anchor_and_absolute_grippers_until_drained(self) -> None:
        first = self._next_action()
        np.testing.assert_allclose(first[np.r_[0:7, 8:15]], 0.6)
        np.testing.assert_allclose(first[[7, 15]], 0.04)
        self.observation.update(dict.fromkeys(ACTION_NAMES, 1.5))
        for offset in (0.2, 0.3):
            action = self._next_action()
            np.testing.assert_allclose(action[np.r_[0:7, 8:15]], 0.5 + offset)
            np.testing.assert_allclose(action[[7, 15]], 0.04)
        self.assertEqual(self.predictions, 1)
        self.robot.hold.assert_called_once()
        fresh = self._next_action()
        np.testing.assert_allclose(fresh[np.r_[0:7, 8:15]], 1.6)
        self.assertEqual(self.predictions, 2)
        self.assertEqual(self.robot.hold.call_count, 2)

    def test_single_step_dispatches_one_action_and_discards_the_rest(self) -> None:
        for _ in range(2):
            _, action, _ = dispatch_single_step(self.engine, self.robot, self.cfg, self.policy)
            np.testing.assert_allclose(action[np.r_[0:7, 8:15]], 0.6)
            self.assertEqual(len(self.queue), 0)
        self.assertEqual(self.predictions, 2)
        self.assertEqual(self.robot.send_action.call_count, 2)

    def test_real_pi05_queue_replans_after_five_actions_from_a_fifty_step_prediction(self) -> None:
        import torch

        from lerobot.policies.pi05.modeling_pi05 import PI05Policy

        self.queue = deque()
        self.policy._action_queue = self.queue
        self.policy.reset = self.queue.clear
        self.policy.eval = lambda: self.policy
        self.policy._rtc_enabled = lambda: False
        self.policy.config.n_action_steps = 5
        self.policy.config.use_visual_memory = False
        self.policy.config.use_proprioceptive_memory = False
        predicted = torch.arange(1, 51, dtype=torch.float32).view(1, 50, 1).expand(1, 50, 16).clone() / 10
        predicted[:, :, [7, 15]] = 0.04
        self.policy.predict_action_chunk = MagicMock(side_effect=lambda batch: predicted.clone())
        self.policy.select_action = MethodType(PI05Policy.select_action, self.policy)
        for index in range(5):
            action = self._next_action()
            np.testing.assert_allclose(action[np.r_[0:7, 8:15]], 0.5 + (index + 1) / 10)
            self.assertEqual(self.policy.count_queued_actions(), 4 - index)
        self.policy.predict_action_chunk.assert_called_once()
        replanned = self._next_action()
        np.testing.assert_allclose(replanned[np.r_[0:7, 8:15]], 0.6)
        self.assertEqual(self.policy.predict_action_chunk.call_count, 2)
        self.assertEqual(self.policy.count_queued_actions(), 4)

    def test_slow_prediction_is_rejected_before_sending(self) -> None:
        self.prediction_delay = self.cfg["max_observation_age_s"] + 1
        with self.assertRaisesRegex(TimeoutError, "deadline"):
            dispatch_single_step(self.engine, self.robot, self.cfg, self.policy)
        self.robot.send_action.assert_not_called()
        self.assertEqual(len(self.queue), 0)

    def test_stale_observation_is_rejected_before_inference(self) -> None:
        observation = self.robot.get_observation()
        self.clock[0] += self.cfg["max_observation_age_s"] + 1
        with self.assertRaisesRegex(TimeoutError, "observation is too old"):
            next_policy_action(self.engine, self.robot, self.cfg, observation, self.policy)
        self.assertEqual(self.predictions, 0)
        self.robot.send_action.assert_not_called()

    def test_camera_failure_during_prediction_prevents_dispatch(self) -> None:
        def capture() -> dict:
            if self.predictions:
                raise CameraFreshnessError("camera stopped during prediction")
            self.robot.observation_time = self.clock[0]
            return self.observation.copy()

        self.robot.get_observation.side_effect = capture
        with self.assertRaisesRegex(CameraFreshnessError, "camera stopped"):
            dispatch_single_step(self.engine, self.robot, self.cfg, self.policy)
        self.robot.send_action.assert_not_called()
        self.assertEqual(len(self.queue), 0)

    def _execute(
        self,
        run_mode: str = "continuous",
        answers: list[str] | None = None,
        *,
        home_error: Exception | None = None,
    ) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("flexiv_deploy_hardware.FlexivDualArm", return_value=self.robot),
            patch("lerobot.inference.SyncInferenceEngine", return_value=self.engine),
            patch("deploy_dualarm_no_rtc.align_deployment_start_pose", side_effect=home_error),
            patch(
                "deploy_dualarm_no_rtc.time.sleep",
                side_effect=lambda seconds: self.clock.__setitem__(0, self.clock[0] + seconds),
            ),
            patch("builtins.input", side_effect=answers if answers is not None else [""]),
        ):
            self.cfg["telemetry"] = str(Path(directory) / "telemetry.jsonl")
            execute_deployment(
                self.cfg, self.policy, None, None, run_mode, (np.zeros(16), {"scene": "scene_0001"})
            )

    def test_cancel_at_home_prevents_policy_start(self) -> None:
        self._execute(answers=["q"])
        self.assertEqual(self.predictions, 0)
        self.robot.send_action.assert_not_called()
        self.robot.wait_for_fresh_observation.assert_not_called()
        self.robot.close.assert_called_once()

    def test_failed_home_prevents_prediction_and_closes_devices(self) -> None:
        with self.assertRaisesRegex(TimeoutError, "home failed"):
            self._execute(home_error=TimeoutError("home failed"))
        self.assertEqual(self.predictions, 0)
        self.robot.send_action.assert_not_called()
        self.robot.wait_for_fresh_observation.assert_not_called()
        self.robot.close.assert_called_once()

    def test_step_execution_waits_for_enter_and_only_sends_one_action(self) -> None:
        with redirect_stdout(io.StringIO()):
            self._execute(run_mode="step", answers=["", "q"])
        self.assertEqual(self.predictions, 1)
        self.robot.send_action.assert_called_once()
        self.robot.wait_for_target.assert_called_once()
        self.assertEqual(len(self.queue), 0)
        self.robot.close.assert_called_once()

    def test_continuous_mode_plays_the_chunk_without_catching_up_after_inference(self) -> None:
        self.cfg["duration_s"] = 0.5
        self.prediction_delay = 0.12
        dispatch_times: list[float] = []

        def send(action: np.ndarray) -> np.ndarray:
            dispatch_times.append(self.clock[0])
            return action

        self.robot.send_action.side_effect = send
        self._execute()
        self.assertEqual(self.predictions, 2)
        self.assertEqual(len(dispatch_times), 4)
        self.assertTrue(np.all(np.diff(dispatch_times) >= 1 / self.cfg["fps"] - 1e-9))
        self.robot.close.assert_called_once()

    def test_continuous_mode_does_not_send_if_duration_expires_during_inference(self) -> None:
        self.cfg["duration_s"] = 0.1
        self.prediction_delay = 0.2
        self._execute()
        self.robot.send_action.assert_not_called()
        self.robot.close.assert_called_once()

    def test_recording_keeps_original_observation_and_drains_after_robot_stop(self) -> None:
        self.cfg["duration_s"] = 0.3
        self.prediction_delay = 0.12
        events: list[str] = []
        captures: list[tuple[dict, dict, float]] = []
        writer = MagicMock()
        writer.record.side_effect = lambda sample, obs, stamp: captures.append((sample, obs, stamp))
        writer.close.side_effect = lambda: events.append("drain")
        self.robot.close.side_effect = lambda: events.append("stop_robot")

        def capture() -> dict:
            now = self.clock[0]
            self.robot.observation_time = now
            self.robot.camera_ages_s = {"image": now}
            return {**self.observation, "_capture_metadata": {"state_read_monotonic_s": now}}

        self.robot.get_observation.side_effect = capture
        with patch("deploy_dualarm_no_rtc.DeploymentLog", return_value=writer):
            self._execute()
        self.assertTrue(captures)
        for sample, obs, stamp in captures:
            self.assertEqual(stamp, obs["_capture_metadata"]["state_read_monotonic_s"])
            self.assertEqual(sample["camera_ages_s"]["image"], stamp)
        self.assertEqual(events, ["stop_robot", "drain"])
