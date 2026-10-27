"""Offline reference, camera lifecycle and desktop tests; never use real hardware."""

from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from align import SCENES, AlignmentWindow, ReferenceStore, TopCamera, fresh_frame, main
from PIL import Image


class TestReferenceStore(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.camera = {"serial": "fake-top", "width": 640, "height": 360, "fps": 15}
        self.store = ReferenceStore(self.root / "references", self.camera)
        self.rgb = np.full((360, 640, 3), 100, dtype=np.uint8)

    def save(self, name: str = "test_01") -> Path:
        return self.store.save(name, self.rgb, 42.0, "offline fixture")

    def test_ten_references_roundtrip_and_resume_without_overwrite(self) -> None:
        for name in SCENES:
            self.save(name)
        resumed = ReferenceStore(self.store.directory, self.camera)
        self.assertEqual(resumed.saved(), list(SCENES))
        for name in SCENES:
            meta, rgb = resumed.load(name)
            self.assertEqual(meta["scene"], name)
            self.assertEqual(meta["note"], "offline fixture")
            np.testing.assert_array_equal(rgb, self.rgb)
        with self.assertRaises(FileExistsError):
            self.save()
        with self.assertRaises(ValueError):
            self.save("../escape")

    def test_reject_camera_or_resolution_change_and_corrupt_image(self) -> None:
        path = self.save()
        for key, value in (("serial", "other"), ("width", 1280), ("fps", 30)):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "differs"):
                ReferenceStore(self.store.directory, {**self.camera, key: value}).load("test_01")
        (path / "top.png").write_bytes(b"bad png")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.store.load("test_01")

    def test_reject_wrong_image_shape_before_creating_reference(self) -> None:
        with self.assertRaises(ValueError):
            self.store.save("test_01", np.zeros((224, 224, 3), dtype=np.uint8), 42.0, "")
        self.assertFalse(self.store.directory.exists())

    def test_confirmation_records_actual_and_reference_without_overwriting(self) -> None:
        self.save()
        reference = self.store.load("test_01")
        actual = self.rgb.copy()
        actual[:, :320] = 200
        first = self.store.save_confirmation("test_01", reference, actual, 43.0, "50k trial 1")
        second = self.store.save_confirmation("test_01", reference, actual, 44.0, "50k trial 2")
        self.assertNotEqual(first, second)
        audit = json.loads((first / "alignment.json").read_text())
        self.assertFalse(audit["robot_state_verified"])
        self.assertFalse(audit["deployment_started"])
        self.assertEqual(audit["reference"], reference[0])
        with Image.open(first / "actual_top.png") as image:
            np.testing.assert_array_equal(np.asarray(image), actual)
        with Image.open(first / "overlay.png") as image:
            self.assertEqual(image.getpixel((10, 10)), (150, 150, 150))

    def test_list_does_not_construct_camera_or_window(self) -> None:
        with (
            patch("sys.argv", ["align.py", "--list", "--directory", str(self.store.directory)]),
            patch("align.TopCamera") as camera,
            patch("align.AlignmentWindow") as window,
        ):
            main()
        camera.assert_not_called()
        window.assert_not_called()

    def test_camera_closed_after_window_error(self) -> None:
        with (
            patch("sys.argv", ["align.py", "--capture", "--directory", str(self.store.directory)]),
            patch("align.TopCamera") as camera,
            patch("align.AlignmentWindow") as window,
        ):
            window.return_value.run.side_effect = RuntimeError("preview failed")
            with self.assertRaisesRegex(RuntimeError, "preview failed"):
                main()
        camera.return_value.close.assert_called_once()
        window.return_value.close.assert_called_once()

    def test_stale_and_future_frames_cannot_pass(self) -> None:
        with patch("align.time.monotonic", return_value=10.0):
            self.assertTrue(fresh_frame((self.rgb, 9.9)))
            self.assertFalse(fresh_frame((self.rgb, 9.0)))
            self.assertFalse(fresh_frame((self.rgb, 11.0)))
            self.assertFalse(fresh_frame(None))

    def test_camera_setup_only_opens_requested_rgb_device_and_releases(self) -> None:
        rs = MagicMock()
        with (
            patch.dict("sys.modules", {"pyrealsense2": rs}),
            patch("align.Thread") as thread,
        ):
            camera = TopCamera(self.camera)
            rs.config.return_value.enable_device.assert_called_once_with("fake-top")
            rs.config.return_value.enable_stream.assert_called_once_with(
                rs.stream.color, 640, 360, rs.format.rgb8, 15
            )
            camera.close()
            camera.close()
            rs.pipeline.return_value.stop.assert_called_once()
            thread.return_value.join.assert_called_once()

    @unittest.skipUnless(os.environ.get("DISPLAY"), "Tk desktop required")
    def test_gui_capture_and_confirmation_require_checkbox_and_fresh_display(self) -> None:
        window = AlignmentWindow(self.store, capture=True, scene=None, note="capture fixture")
        self.addCleanup(window.close)
        camera = MagicMock()
        camera.latest.return_value = self.rgb, time.monotonic()
        window.camera = camera
        window.update()
        window.root.update_idletasks()
        window.confirm()
        self.assertEqual(self.store.saved(), [])
        camera.latest.return_value = self.rgb, time.monotonic()
        window.update()
        window.confirm_visual.set(True)
        window.confirm()
        self.assertEqual(self.store.saved(), ["test_01"])
        self.assertEqual(window.scene.get(), "test_02")
        self.assertFalse(window.confirm_visual.get())
        window.close()

        compare = AlignmentWindow(self.store, capture=False, scene="test_01", note="trial")
        self.addCleanup(compare.close)
        compare.camera = camera
        camera.latest.return_value = self.rgb, time.monotonic()
        compare.update()
        compare.confirm_visual.set(True)
        compare.displayed = self.rgb, time.monotonic() - 1
        compare.confirm()
        self.assertIsNone(compare.result)
        camera.latest.return_value = self.rgb, time.monotonic()
        compare.update()
        compare.confirm()
        self.assertIsNotNone(compare.result)
        self.assertIsNone(compare.error)


if __name__ == "__main__":
    unittest.main()
