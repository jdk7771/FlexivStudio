"""Regression checks for timestamp alignment and the exported action convention."""

import argparse
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import sleep
from unittest.mock import patch

import h5py
import numpy as np
from PIL import Image
from port_flexiv import (
    Layout,
    ScenePlan,
    contiguous_segments,
    make_labels,
    nearest_indices,
    parallel_frames,
    plan_scene,
    read_rgb,
)


class TestFlexivConversion(unittest.TestCase):
    def test_parallel_prefetch_keeps_images_and_labels_in_order(self):
        def fake_rgb(path, size):
            value = int(path.stem)
            if value == 0:
                sleep(0.02)
            return np.full((size, size, 3), value, dtype=np.uint8)

        plan = ScenePlan(
            Path("scene"),
            np.arange(4),
            np.repeat(np.arange(4)[:, None], 16, axis=1),
            np.repeat((np.arange(4) + 10)[:, None], 16, axis=1),
            {key: [Path(f"{i}.png") for i in range(4)] for key in ["global", "wrist_left", "wrist_right"]},
            [np.arange(4)],
            {},
        )
        with patch("port_flexiv.read_rgb", side_effect=fake_rgb), ThreadPoolExecutor(4) as pool:
            frames = list(parallel_frames(pool, plan, np.arange(4), 8, "fold towel", 2))
        for i, frame in enumerate(frames):
            np.testing.assert_array_equal(frame["observation.images.image"], i)
            np.testing.assert_array_equal(frame["observation.state"], i)
            np.testing.assert_array_equal(frame["action"], i + 10)

    def test_next_joint_target_and_causal_absolute_gripper(self):
        timestamps = np.array([0, 100, 200, 300], dtype=float)
        signals = {}
        arms = [("left", "L"), ("right", "R")]
        for offset, (_, serial) in enumerate(arms):
            signals[f"joint_position_rad_{serial}"] = np.repeat(
                (timestamps / 1000 + offset)[:, None], 7, axis=1
            )
            signals[f"ee_state_{serial}"] = np.array([[0.02], [0.03], [0.04], [0.05]])
            signals[f"ee_command_{serial}"] = np.array([[0.1], [0], [0.1], [0]])
        states, actions = make_labels(timestamps, signals, arms, np.array([50, 150]), 100)
        np.testing.assert_allclose(states[:, 0], [0.05, 0.15])
        np.testing.assert_allclose(actions[:, 0], [0.15, 0.25])
        np.testing.assert_allclose(actions[:, 8], [1.15, 1.25])
        np.testing.assert_allclose(states[:, 7], [0.025, 0.035])
        # Future commands must not leak backwards or get interpolated/subtracted from state.
        np.testing.assert_allclose(actions[:, [7, 15]], [[0.1, 0.1], [0, 0]])
        # Both chunk targets use the SAME initial observation as their delta origin.
        np.testing.assert_allclose(actions[:, 0] - states[0, 0], [0.1, 0.2], atol=1e-7)

    def test_nearest_image_and_tie(self):
        indices = nearest_indices(np.array([100, 200, 400]), np.array([90, 150, 260, 450]))
        np.testing.assert_array_equal(indices, [0, 0, 1, 2])

    def test_a_missing_frame_cannot_join_two_action_chunks(self):
        segments = contiguous_segments(np.array([1, 1, 1, 0, 1, 1, 1, 1, 0, 1], dtype=bool), 3)
        self.assertEqual([segment.tolist() for segment in segments], [[0, 1, 2], [4, 5, 6, 7]])

    def test_duplicate_timestamp_uses_last_record_and_splits_camera_outage(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene = Path(tmp)
            (scene / "lowdim").mkdir()
            timestamps = np.array([0, 100, 100, 200, 300, 400, 500, 600, 700, 800])
            with h5py.File(scene / "lowdim/lowdim.h5", "w") as lowdim:
                lowdim["timestamp"] = timestamps
                for serial in ("L", "R"):
                    joints = np.repeat((timestamps / 1000)[:, None], 7, axis=1)
                    joints[1] = 99  # Stale row at the duplicated millisecond must be removed.
                    lowdim[f"joint_position_rad_{serial}"] = joints
                    lowdim[f"ee_state_{serial}"] = np.zeros((len(timestamps), 1))
                    lowdim[f"ee_command_{serial}"] = np.full((len(timestamps), 1), 0.1)
            for serial in ("C1", "C2", "C3"):
                camera = scene / f"cam_{serial}" / "color"
                camera.mkdir(parents=True)
                for time in (0, 100, 200, 400, 500, 600, 700):
                    (camera / f"{time}.png").touch()
            layout = Layout(
                [("left", "L"), ("right", "R")], {"global": "C1", "wrist_left": "C2", "wrist_right": "C3"}
            )
            args = argparse.Namespace(
                fps=10, chunk_size=3, max_camera_skew_ms=40, max_lowdim_gap_ms=150, max_frames=None
            )
            plan = plan_scene(scene, layout, args)
            self.assertEqual([segment.tolist() for segment in plan.segments], [[0, 1, 2], [4, 5, 6, 7]])
            self.assertEqual(plan.report["duplicate_lowdim_timestamps"], 1)
            np.testing.assert_allclose(plan.states[1, 0], 0.1)
            np.testing.assert_allclose(plan.actions[-1, 0], 0.8)
            self.assertEqual(layout.names[7], "left_gripper")
            self.assertEqual(layout.names[15], "right_gripper")

    def test_rgb_letterbox_preserves_aspect_ratio(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "image.png"
            Image.new("RGB", (80, 40), color=(255, 0, 0)).save(path)
            rgb = read_rgb(path, 32)
            self.assertEqual(rgb.shape, (32, 32, 3))
            self.assertEqual(rgb.dtype, np.uint8)
            np.testing.assert_array_equal(rgb[0], 0)
            np.testing.assert_array_equal(rgb[16, 16], [255, 0, 0])


if __name__ == "__main__":
    unittest.main()
