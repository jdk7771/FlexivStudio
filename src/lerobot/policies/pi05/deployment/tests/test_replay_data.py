"""Recorded follower replay checks. All device interactions are mocked."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
from flexiv_deploy_contract import ACTION_NAMES, read_json
from replay_data import (
    Recording,
    available_scenes,
    choose_scene,
    execute_replay,
    load_manifest,
    load_recording,
    main,
    replay_targets,
    validate_speed,
    wait_for_start,
)


class TestRecordingFiles(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.source = self.root / "scene_0001/lowdim/lowdim.h5"
        self.source.parent.mkdir(parents=True)
        self.cfg = read_json(Path(__file__).resolve().parents[1] / "configs/deploy_config.json")
        self.manifest = {
            "source": str(self.root),
            "dimension_names": ACTION_NAMES,
            "units": {"joint": "radians", "gripper": "meters"},
            "arms": {"left": "063906", "right": "062106"},
            "max_lowdim_gap_ms": 50,
            "scenes": [{"scene": "scene_0001"}],
        }
        self.manifest_path = self.root / "manifest.json"
        self.manifest_path.write_text(json.dumps(self.manifest))
        self.cfg["start_pose"]["manifest"] = str(self.manifest_path)
        with h5py.File(self.source, "w") as lowdim:
            lowdim["timestamp"] = [1000, 1000, 1025, 1050, 1075, 1100]
            for serial, offset in (("063906", 0.0), ("062106", 1.0)):
                lowdim[f"joint_position_rad_{serial}"] = np.tile(
                    np.arange(6).reshape(-1, 1) * 0.01 + offset, (1, 7)
                )
                lowdim[f"ee_command_{serial}"] = np.arange(6).reshape(-1, 1) * 0.02
                # Measured widths differ deliberately: replay must use command history.
                lowdim[f"ee_state_{serial}"] = np.zeros((6, 1))

    def load(self) -> Recording:
        return load_recording("scene_0001", self.source, self.manifest, self.cfg)

    def test_duplicate_timestamps_keep_last_and_grippers_use_previous_command(self) -> None:
        recording = self.load()
        np.testing.assert_allclose(recording.timestamps_s, [0.0, 1 / 15, 0.1])
        np.testing.assert_allclose(recording.targets[0, :7], 0.01)
        np.testing.assert_allclose(recording.targets[0, 8:15], 1.01)
        np.testing.assert_allclose(recording.targets[:, [7, 15]], [[0.02] * 2, [0.06] * 2, [0.1] * 2])
        np.testing.assert_allclose(recording.targets[-1, :7], 0.05)
        self.assertEqual(recording.source_start_ms, 1000.0)

    def test_missing_gripper_command_is_not_replaced_with_measured_width(self) -> None:
        with h5py.File(self.source, "a") as lowdim:
            del lowdim["ee_command_063906"]
        with self.assertRaises(KeyError):
            self.load()

    def test_bad_timestamps_and_large_gaps_are_rejected(self) -> None:
        for timestamps in (
            [1000] * 6,
            [1000, 1001, 999, 1003, 1004, 1005],
            [1000, 1001, 1002, 1200, 1201, 1202],
        ):
            with self.subTest(timestamps=timestamps):
                with h5py.File(self.source, "a") as lowdim:
                    lowdim["timestamp"][:] = timestamps
                with self.assertRaises(ValueError):
                    self.load()

    def test_nonfinite_joints_and_invalid_gripper_units_are_rejected(self) -> None:
        with h5py.File(self.source, "a") as lowdim:
            lowdim["joint_position_rad_063906"][0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "joint_position_rad"):
            self.load()
        with h5py.File(self.source, "a") as lowdim:
            lowdim["joint_position_rad_063906"][0, 0] = 0
            lowdim["ee_command_063906"][0, 0] = 100
        with self.assertRaisesRegex(ValueError, "meters"):
            self.load()

    def test_manifest_serial_mismatch_rejected(self) -> None:
        self.cfg["hardware"]["arms"]["left"]["serial"] = "Rizon4-063848"
        with self.assertRaisesRegex(ValueError, "left follower serial"):
            load_manifest(self.cfg)

    def test_scene_menu_accepts_number_or_name_and_can_cancel(self) -> None:
        choices = available_scenes(self.manifest)
        self.assertEqual(choices, [("scene_0001", self.source, 0.1)])
        for answer, expected in (("1", "scene_0001"), ("scene_0001", "scene_0001"), ("q", None)):
            with patch("builtins.input", return_value=answer), redirect_stdout(io.StringIO()):
                self.assertEqual(choose_scene(choices), expected)

    def test_check_and_list_never_construct_hardware_or_require_sdk(self) -> None:
        config = self.root / "config.json"
        config.write_text(json.dumps(self.cfg))
        for flags in (["--list"], ["--scene", "scene_0001", "--check"]):
            with (
                patch("sys.argv", ["replay_data.py", "--config", str(config), *flags]),
                patch("replay_data.FlexivDualArm") as hardware,
                patch("replay_data.require_compatible_rdk") as sdk,
                redirect_stdout(io.StringIO()),
            ):
                main()
            hardware.assert_not_called()
            sdk.assert_not_called()


class TestReplayExecution(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = read_json(Path(__file__).resolve().parents[1] / "configs/deploy_config.json")
        targets = np.zeros((3, 16))
        targets[1, [0, 8]] = 0.01
        targets[2, [0, 8]] = 0.02
        self.recording = Recording(
            "scene_0002", Path("lowdim.h5"), np.array([0.0, 0.1, 0.2]), targets, 1000.0
        )
        self.robot = MagicMock()
        for side in ("left", "right"):
            self.robot.robots[side].info.return_value = SimpleNamespace(q_min=[-3] * 7, q_max=[3] * 7)
        self.enterContext(patch("replay_data.FlexivDualArm", return_value=self.robot))
        self.enterContext(patch("replay_data.require_compatible_rdk"))
        self.align = self.enterContext(patch("replay_data.align_start_pose"))

    def test_align_then_key_then_replay_and_no_cameras(self) -> None:
        def press_start() -> bool:
            self.align.assert_called_once()
            self.robot.hold.assert_called_once()
            self.robot.send_action.assert_not_called()
            return True

        with (
            patch("replay_data.wait_for_start", side_effect=press_start),
            patch("replay_data.replay_targets") as replay,
        ):
            execute_replay(self.cfg, self.recording, 1.0)
        self.robot.connect.assert_called_once_with(with_cameras=False)
        np.testing.assert_array_equal(self.align.call_args.args[1][0], self.recording.targets[0])
        self.assertEqual(self.align.call_args.args[1][1]["scene"], "scene_0002")
        self.robot.check_initial_pose.assert_called_once()
        replay.assert_called_once_with(self.robot, self.recording, self.cfg, 1.0)
        self.robot.close.assert_called_once()

    def test_cancel_or_interrupt_closes_without_playing(self) -> None:
        for interrupt in (False, True):
            self.robot.reset_mock()
            with (
                patch(
                    "replay_data.wait_for_start",
                    return_value=False,
                    side_effect=KeyboardInterrupt if interrupt else None,
                ),
                patch("replay_data.replay_targets") as replay,
            ):
                if interrupt:
                    with self.assertRaises(KeyboardInterrupt):
                        execute_replay(self.cfg, self.recording, 1.0)
                else:
                    execute_replay(self.cfg, self.recording, 1.0)
            replay.assert_not_called()
            self.robot.close.assert_called_once()

    def test_joint_limit_violation_anywhere_prevents_alignment(self) -> None:
        self.recording.targets[-1, 8] = 4.0
        with self.assertRaisesRegex(ValueError, "right joints exceed"):
            execute_replay(self.cfg, self.recording, 1.0)
        self.align.assert_not_called()
        self.robot.send_action.assert_not_called()
        self.robot.close.assert_called_once()

    def test_half_speed_preserves_targets_and_stretches_intervals(self) -> None:
        now = [0.0]
        sent_at = []
        self.robot.send_action.side_effect = lambda target: sent_at.append(now[0])

        def sleep(seconds: float) -> None:
            now[0] += seconds

        with (
            patch("replay_data.time.monotonic", side_effect=lambda: now[0]),
            patch("replay_data.time.sleep", side_effect=sleep),
        ):
            replay_targets(self.robot, self.recording, self.cfg, 0.5)
        np.testing.assert_allclose(sent_at, [0.0, 0.2, 0.4])
        for invocation, expected in zip(
            self.robot.send_action.call_args_list, self.recording.targets, strict=True
        ):
            np.testing.assert_array_equal(invocation.args[0], expected)
        self.robot.wait_for_target.assert_called_once()

    def test_late_command_never_causes_catchup_burst_and_failures_propagate(self) -> None:
        now = [0.0]
        sent_at = []

        def send(target: np.ndarray) -> None:
            sent_at.append(now[0])
            now[0] += 0.25

        self.robot.send_action.side_effect = send
        with (
            patch("replay_data.time.monotonic", side_effect=lambda: now[0]),
            patch("replay_data.time.sleep") as sleep,
        ):
            replay_targets(self.robot, self.recording, self.cfg, 1.0)
        sleep.assert_not_called()
        np.testing.assert_allclose(sent_at, [0.0, 0.25, 0.5])
        with (
            patch("replay_data.wait_for_start", return_value=True),
            patch("replay_data.replay_targets", side_effect=ValueError("target too far")),
            self.assertRaisesRegex(ValueError, "target too far"),
        ):
            execute_replay(self.cfg, self.recording, 1.0)
        self.robot.close.assert_called_once()

    def test_only_enter_starts_and_invalid_speed_rejected(self) -> None:
        for answer in ("q", "quit"):
            with patch("builtins.input", return_value=answer):
                self.assertFalse(wait_for_start())
        with patch("builtins.input", side_effect=EOFError):
            self.assertFalse(wait_for_start())
        with patch("builtins.input", side_effect=["later", ""]), redirect_stdout(io.StringIO()):
            self.assertTrue(wait_for_start())
        for speed in (0, -1, 1.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                validate_speed(speed)


if __name__ == "__main__":
    unittest.main()
