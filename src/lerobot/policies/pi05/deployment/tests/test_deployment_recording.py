"""Offline recorder integration, bounded overflow, and writer-failure tests."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
import unittest
from pathlib import Path
from queue import Empty
from unittest.mock import MagicMock, patch

import numpy as np
from deployment_recording import SLOTS, DeploymentLog, add_recording_arguments, configure_recording
from flexiv_deploy_contract import CAMERA_NAMES
from PIL import Image


class TestDeploymentRecording(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.cfg: dict = {"telemetry": str(self.root / "telemetry.jsonl"), "record": True}
        self.observation = {name: np.full((224, 224, 3), 80, dtype=np.uint8) for name in CAMERA_NAMES}
        self.observation["_capture_metadata"] = {"state_read_monotonic_s": 42.0}
        self.sample = {"state": [0.1] * 16, "applied_absolute_action": [0.2] * 16, "wall_time": 100.0}

    def make_log(self) -> DeploymentLog:
        log = DeploymentLog(self.cfg, "continuous", rtc=True)
        self.addCleanup(log.close)
        return log

    def test_disabled_keeps_telemetry_without_worker_or_images(self) -> None:
        self.cfg["record"] = False
        with patch("deployment_recording.mp.get_context") as context:
            log = self.make_log()
            self.assertTrue(log.record(self.sample, {}, None))
            log.close()
        context.assert_not_called()
        self.assertFalse((self.root / "recordings").exists())
        self.assertEqual(json.loads((self.root / "telemetry.jsonl").read_text()), self.sample)

    def test_images_metadata_and_state_stay_aligned_and_own_their_pixels(self) -> None:
        log = self.make_log()
        for index in range(3):
            for name in CAMERA_NAMES:
                self.observation[name].fill(60 + index * 20)
            self.assertTrue(log.record({**self.sample, "step": index}, self.observation, 40.0 + index))
        # The producer can reuse/modify images immediately after returning.
        for name in CAMERA_NAMES:
            self.observation[name].fill(255)
        log.close()
        assert log.root is not None
        records = [json.loads(line) for line in (log.root / "frames.jsonl").read_text().splitlines()]
        self.assertEqual(len(records), 3)
        for index, record in enumerate(records):
            self.assertEqual(record["sequence"], index)
            self.assertEqual(record["telemetry"]["step"], index)
            self.assertEqual(record["telemetry"]["state"], self.sample["state"])
            self.assertEqual(record["observation_time_monotonic_s"], 40.0 + index)
            self.assertEqual(record["capture"]["state_read_monotonic_s"], 42.0)
            for relative in record["images"].values():
                with Image.open(log.root / relative) as rgb:
                    self.assertEqual(rgb.size, (224, 224))
                    self.assertEqual(rgb.getpixel((112, 112)), (60 + index * 20,) * 3)
        summary = json.loads((log.root / "summary.json").read_text())
        self.assertEqual(summary["written"], 3)
        self.assertEqual(summary["unwritten_submissions"], 0)
        self.assertIsNone(summary["error"])

    def test_full_buffer_drops_without_waiting_or_copying_images(self) -> None:
        log = self.make_log()
        free, released, jobs = log._free, log._released, log._jobs
        try:
            log._free = []
            log._released = MagicMock()
            log._released.get_nowait.side_effect = Empty
            log._jobs = MagicMock()
            with patch("deployment_recording.np.copyto") as copy:
                started = time.perf_counter()
                for _ in range(100):
                    self.assertFalse(log.record(self.sample, {}, 42.0))
                self.assertLess(time.perf_counter() - started, 0.2)
            copy.assert_not_called()
            log._jobs.put.assert_not_called()
            log._jobs.put_nowait.assert_not_called()
            self.assertEqual(log.dropped, 100)
            self.assertEqual(log._buffers.shape[0], SLOTS)
        finally:
            log._free, log._released, log._jobs = free, released, jobs

    def test_disk_error_stops_only_writer_and_marks_incomplete_recording(self) -> None:
        log = self.make_log()
        assert log.root is not None
        directory = log.root / "images" / CAMERA_NAMES[0]
        directory.rmdir()
        directory.write_text("simulate an unwritable image directory")
        self.assertTrue(log.record(self.sample, self.observation, 42.0))
        log._process.join(timeout=5)
        self.assertFalse(log._process.is_alive())
        self.assertFalse(log.record(self.sample, self.observation, 43.0))
        log.close()
        summary = json.loads((log.root / "summary.json").read_text())
        self.assertIn("NotADirectoryError", summary["error"])
        self.assertEqual(summary["unwritten_submissions"], 1)
        self.assertEqual(summary["dropped_at_submit"], 1)
        self.assertEqual((log.root / "frames.jsonl").read_text(), "")

    def test_cli_default_off_and_explicit_enable_disable(self) -> None:
        parser = argparse.ArgumentParser()
        add_recording_arguments(parser)
        for flags, enabled in (([], False), (["--record"], True), (["--record", "--no-record"], False)):
            with self.subTest(flags=flags):
                cfg = self.cfg.copy()
                configure_recording(cfg, parser.parse_args(flags))
                self.assertEqual(cfg["record"], enabled)
                self.assertEqual(Path(cfg["record_dir"]), self.root / "recordings")


if __name__ == "__main__":
    unittest.main()
