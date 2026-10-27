"""Offline high-rate polling, timestamp differentiation and lifecycle tests."""

from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import deploy_dualarm
import flexiv_deploy_hardware
from deploy_rtc_states import main, recording_robot_class
from high_rate_state_recording import StateRecorder


class TestStateRecorder(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.recorder = StateRecorder(self.root / "run", {}, 200)
        self.addCleanup(self.recorder.close)
        self.state = SimpleNamespace(
            timestamp=[100, 0],
            q=[0.0] * 7,
            dq=[1.0] * 7,
            tau=[2.0] * 7,
            tau_ext=[3.0] * 7,
            tcp_pose=[0.0] * 7,
            tcp_vel=[0.0] * 6,
            ext_wrench_in_tcp=[0.0] * 6,
        )
        self.arm = SimpleNamespace(states=lambda: self.state)
        self.gripper = SimpleNamespace(states=lambda: SimpleNamespace(width=0.04, force=2.0, is_moving=False))
        self.arms = dict.fromkeys(("left", "right"), self.arm)
        self.grippers = dict.fromkeys(("left", "right"), self.gripper)

    def test_acceleration_uses_distinct_robot_times_and_skips_duplicates_and_gaps(self) -> None:
        first = self.recorder.sample(self.arms, self.grippers)["left"]
        self.assertIsNone(first["ddq_estimate"])
        self.state.timestamp = [100, 5_000_000]
        self.state.dq = [1.01] * 7
        second = self.recorder.sample(self.arms, self.grippers)["left"]
        self.assertAlmostEqual(second["ddq_estimate"][0], 2.0)
        self.assertEqual(second["dq"], [1.01] * 7)
        duplicate = self.recorder.sample(self.arms, self.grippers)["left"]
        self.assertTrue(duplicate["duplicate_timestamp"])
        self.assertIsNone(duplicate["ddq_estimate"])
        self.state.timestamp = [101, 0]
        self.assertIsNone(self.recorder.sample(self.arms, self.grippers)["left"]["ddq_estimate"])
        self.state.timestamp = [99, 0]
        self.assertIsNone(self.recorder.sample(self.arms, self.grippers)["left"]["ddq_estimate"])

    def test_actual_background_polling_and_drain_without_images(self) -> None:
        self.recorder.start(self.arms, self.grippers)
        time.sleep(0.15)
        self.recorder.close()
        rows = [
            json.loads(line) for line in (self.recorder.directory / "states.jsonl").read_text().splitlines()
        ]
        self.assertGreater(len(rows), 5)
        self.assertEqual(rows[0]["left"]["gripper"]["width"], 0.04)
        self.assertEqual([row["sequence"] for row in rows], list(range(len(rows))))
        summary = json.loads((self.recorder.directory / "summary.json").read_text())
        self.assertEqual(summary["samples"], summary["written"])
        self.assertEqual(summary["distinct_robot_timestamps"], {"left": 1, "right": 1})
        self.assertIsNone(summary["error"])
        self.assertFalse((self.recorder.directory / "images").exists())

    def test_full_queue_drops_immediately(self) -> None:
        for i in range(2048):
            self.recorder.samples += 1
            self.recorder.submit({"sequence": i})
        started = time.perf_counter()
        self.recorder.samples += 1
        self.recorder.submit({"sequence": 2048})
        self.assertLess(time.perf_counter() - started, 0.1)
        self.assertEqual(self.recorder.dropped, 1)
        self.assertEqual(self.recorder._queue.qsize(), 2048)

    def test_writer_failure_stops_recording_and_keeps_error(self) -> None:
        self.recorder._stream.close()
        stream = MagicMock()
        stream.write.side_effect = OSError("disk full")
        self.recorder._stream = stream
        self.recorder.start(self.arms, self.grippers)
        self.assertTrue(self.recorder._stop.wait(1))
        self.recorder.close()
        self.assertIn("disk full", self.recorder.error)


class TestStateEntry(unittest.TestCase):
    def test_check_does_not_create_recorder_and_forces_images_off(self) -> None:
        with (
            patch("sys.argv", ["deploy_rtc_states.py", "--state-hz", "500", "--check"]),
            patch.object(deploy_dualarm, "inspect_checkpoint", return_value={"kind": "full"}) as inspect,
            patch.object(deploy_dualarm, "load_policy") as load,
            patch("deploy_rtc_states.StateRecorder") as recorder,
        ):
            main()
        self.assertFalse(inspect.call_args.args[0]["record"])
        recorder.assert_not_called()
        load.assert_not_called()

    def test_stop_motion_precedes_recording_drain(self) -> None:
        order: list[str] = []

        class Base:
            robots: dict = {}
            grippers: dict = {}

            def connect(self, **kwargs: object) -> None:
                order.append("connect")

            def close(self) -> None:
                order.append("stop_robot")

        with (
            patch.object(flexiv_deploy_hardware, "FlexivDualArm", Base),
            patch("deploy_rtc_states.StateRecorder") as recorder,
        ):
            recorder.return_value.start.side_effect = lambda *args: order.append("start_recorder")
            recorder.return_value.request_stop.side_effect = lambda: order.append("signal_recorder")
            recorder.return_value.close.side_effect = lambda: order.append("drain")
            robot = recording_robot_class({}, 200, Path("unused"))()
            robot.connect()
            robot.close()
        self.assertEqual(order, ["connect", "start_recorder", "signal_recorder", "stop_robot", "drain"])

    def test_reject_image_record_option_and_invalid_frequency(self) -> None:
        for options in (["--record"], ["--state-hz", "0"], ["--state-hz", "1001"]):
            with patch("sys.argv", ["deploy_rtc_states.py", *options]), self.assertRaises(SystemExit):
                main()


if __name__ == "__main__":
    unittest.main()
