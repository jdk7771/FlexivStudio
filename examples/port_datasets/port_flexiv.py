#!/usr/bin/env python
"""Convert timestamped Flexiv HDF5 + RGB scenes to LeRobot for relative-action pi0.5.

See README_flexiv.md for installation, action semantics, and training flags.
The source is read-only; the output must be a new directory. Nothing is uploaded.
"""

import argparse
import json
import logging
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml
from PIL import Image, ImageOps

LOGGER = logging.getLogger(__name__)
CAMERA_KEYS = {
    "global": "observation.images.image",
    "wrist_left": "observation.images.image2",
    "wrist_right": "observation.images.image3",
}


@dataclass
class Layout:
    arms: list[tuple[str, str]]
    cameras: dict[str, str]

    @property
    def names(self) -> list[str]:
        return [
            name
            for side, _ in self.arms
            for name in (*[f"{side}_joint_{i}" for i in range(1, 8)], f"{side}_gripper")
        ]


@dataclass
class ScenePlan:
    scene: Path
    times_ms: np.ndarray
    states: np.ndarray
    actions: np.ndarray
    images: dict[str, list[Path]]
    segments: list[np.ndarray]
    report: dict[str, Any]


def read_layout(source: Path) -> Layout:
    with (source / "metadata.yaml").open() as stream:
        meta = yaml.safe_load(stream)
    sides = meta["platform"]["description"]["arms"]
    if sides != ["left", "right"] or len(meta["arms"]) != 2:
        raise ValueError("Expected metadata arms ordered [left, right], with two follower arms.")
    arms = [
        (side, arm["follower_arm"]["sn"].rsplit("-", 1)[-1])
        for side, arm in zip(sides, meta["arms"], strict=True)
    ]
    serials = {camera["name"]: str(camera["sn"]) for camera in meta["cameras"]}
    cameras = {name: serials[name] for name in CAMERA_KEYS}
    return Layout(arms, cameras)


def nearest_indices(timestamps: np.ndarray, queries: np.ndarray) -> np.ndarray:
    right = np.searchsorted(timestamps, queries).clip(0, len(timestamps) - 1)
    left = (right - 1).clip(0)
    return np.where(abs(timestamps[left] - queries) <= abs(timestamps[right] - queries), left, right)


def interpolate(timestamps: np.ndarray, values: np.ndarray, queries: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [np.interp(queries, timestamps, values[:, i]) for i in range(values.shape[1])]
    ).astype(np.float32)


def make_labels(
    timestamps: np.ndarray,
    signals: dict[str, np.ndarray],
    arms: list[tuple[str, str]],
    times_ms: np.ndarray,
    step_ms: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Store absolute targets; the LeRobot processor computes chunk-relative joint deltas."""
    states, actions = [], []
    command_indices = (np.searchsorted(timestamps, times_ms, side="right") - 1).clip(0)
    for _, serial in arms:
        joints = signals[f"joint_position_rad_{serial}"]
        states.extend(
            [
                interpolate(timestamps, joints, times_ms),
                interpolate(timestamps, signals[f"ee_state_{serial}"], times_ms),
            ]
        )
        actions.extend(
            [
                interpolate(timestamps, joints, times_ms + step_ms),
                signals[f"ee_command_{serial}"][command_indices],
            ]
        )
    return np.concatenate(states, axis=1), np.concatenate(actions, axis=1).astype(np.float32)


def contiguous_segments(valid: np.ndarray, min_frames: int) -> list[np.ndarray]:
    indices = np.flatnonzero(valid)
    if not len(indices):
        return []
    runs = np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1)
    return [run for run in runs if len(run) >= min_frames]


def plan_scene(scene: Path, layout: Layout, args: argparse.Namespace) -> ScenePlan:
    with h5py.File(scene / "lowdim/lowdim.h5", "r") as lowdim:
        timestamps = np.asarray(lowdim["timestamp"], dtype=np.float64).reshape(-1)
        if len(timestamps) < 2 or not np.isfinite(timestamps).all() or (np.diff(timestamps) < 0).any():
            raise ValueError(f"{scene.name}: lowdim timestamps must be finite and nondecreasing.")
        # Millisecond timestamps can repeat; retain the last record at each timestamp.
        keep = np.r_[np.diff(timestamps) != 0, True]
        signals = {}
        for _, serial in layout.arms:
            for prefix, width in (("joint_position_rad", 7), ("ee_state", 1), ("ee_command", 1)):
                key = f"{prefix}_{serial}"
                values = np.asarray(lowdim[key], dtype=np.float32)
                if values.shape != (len(timestamps), width) or not np.isfinite(values).all():
                    raise ValueError(f"{scene.name}: invalid shape/nonfinite values in {key}.")
                signals[key] = values[keep]
        duplicate_count = int((~keep).sum())
        timestamps = timestamps[keep]

    camera_times, camera_paths = {}, {}
    for name, serial in layout.cameras.items():
        paths = list((scene / f"cam_{serial}" / "color").glob("*.png"))
        paths.sort(key=lambda path: int(path.stem))
        if not paths:
            raise ValueError(f"{scene.name}: missing RGB camera {name}, serial {serial}.")
        times = np.array([int(path.stem) for path in paths], dtype=np.float64)
        if len(times) > 1 and (np.diff(times) <= 0).any():
            raise ValueError(f"{scene.name}: camera {name} has duplicate/reversed timestamps.")
        camera_times[name], camera_paths[name] = times, paths

    step_ms = 1000 / args.fps
    start = max(timestamps[0], *(times[0] for times in camera_times.values()))
    stop = min(timestamps[-1] - step_ms, *(times[-1] for times in camera_times.values()))
    if stop < start:
        raise ValueError(f"{scene.name}: no common camera/state/next-action time interval.")
    grid = start + np.arange(int(np.floor((stop - start) / step_ms)) + 1) * step_ms
    valid = np.ones(len(grid), dtype=bool)
    images, camera_report = {}, {}
    for name, times in camera_times.items():
        indices = nearest_indices(times, grid)
        skew = abs(times[indices] - grid)
        valid &= skew <= args.max_camera_skew_ms
        images[name] = [camera_paths[name][index] for index in indices]
        camera_report[name] = {
            "serial": layout.cameras[name],
            "feature": CAMERA_KEYS[name],
            "max_skew_ms_before_filter": float(skew.max()),
            "p95_skew_ms_before_filter": float(np.percentile(skew, 95)),
            "reused_frames_before_filter": int((np.diff(indices) == 0).sum()),
        }
    # Do not interpolate through a lowdim outage, including the next-state target interval.
    for gap_index in np.flatnonzero(np.diff(timestamps) > args.max_lowdim_gap_ms):
        valid &= ~((grid < timestamps[gap_index + 1]) & (grid + step_ms > timestamps[gap_index]))
    segments = contiguous_segments(valid, args.chunk_size)
    if args.max_frames is not None:
        segments = [segment[: args.max_frames] for segment in segments]
    states, actions = make_labels(timestamps, signals, layout.arms, grid, step_ms)
    retained = sum(len(segment) for segment in segments)
    report = {
        "scene": scene.name,
        "grid_frames": len(grid),
        "retained_frames": retained,
        "filtered_frames": len(grid) - retained,
        "episodes": len(segments),
        "duplicate_lowdim_timestamps": duplicate_count,
        "max_lowdim_gap_ms": float(np.diff(timestamps).max()),
        "cameras": camera_report,
        "segments": [
            {
                "source_start_ms": float(grid[segment[0]]),
                "source_end_ms": float(grid[segment[-1]]),
                "frames": len(segment),
            }
            for segment in segments
        ],
    }
    return ScenePlan(scene, grid, states, actions, images, segments, report)


def read_rgb(path: Path, size: int) -> np.ndarray:
    with Image.open(path) as image:
        padded = ImageOps.pad(
            image.convert("RGB"), (size, size), method=Image.Resampling.BILINEAR, color=(0, 0, 0)
        )
        return np.asarray(padded, dtype=np.uint8).copy()


def load_frame(plan: ScenePlan, index: int, image_size: int, task: str) -> dict:
    frame = {
        "observation.state": plan.states[index].copy(),
        "action": plan.actions[index].copy(),
        "task": task,
    }
    for name, key in CAMERA_KEYS.items():
        frame[key] = read_rgb(plan.images[name][index], image_size)
    return frame


def parallel_frames(
    pool: ThreadPoolExecutor,
    plan: ScenePlan,
    segment: np.ndarray,
    image_size: int,
    task: str,
    prefetch_frames: int,
) -> Iterator[dict]:
    # Bounded batches keep decoded image memory bounded; map preserves chronological order.
    for offset in range(0, len(segment), prefetch_frames):
        batch = segment[offset : offset + prefetch_frames]
        yield from pool.map(lambda index: load_frame(plan, index, image_size, task), batch)


def write_dataset(plans: list[ScenePlan], layout: Layout, args: argparse.Namespace, manifest: dict) -> None:
    # Keep the read-only audit usable without the much larger training dependencies.
    from lerobot.configs import RGBEncoderConfig
    from lerobot.datasets.dataset_tools import recompute_stats
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    features = {
        "observation.state": {"dtype": "float32", "shape": (16,), "names": layout.names},
        "action": {"dtype": "float32", "shape": (16,), "names": layout.names},
        **{
            key: {
                "dtype": "video",
                "shape": (args.image_size, args.image_size, 3),
                "names": ["height", "width", "channels"],
            }
            for key in CAMERA_KEYS.values()
        },
    }
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=args.output,
        robot_type="flexiv_dual_arm",
        fps=args.fps,
        features=features,
        use_videos=True,
        video_backend="pyav",
        rgb_encoder=RGBEncoderConfig(vcodec="h264", crf=18, preset="fast"),
        streaming_encoding=True,
        # The recorder drops frames after a 100 ms queue timeout. Offline conversion
        # must preserve every frame, so allow a whole episode to queue during stalls.
        encoder_queue_maxsize=max(len(segment) for plan in plans for segment in plan.segments),
        encoder_threads=2,
    )
    manifest_path = args.output / "meta/conversion_manifest.json"
    manifest["status"] = "writing"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    episode_index = 0
    started = time.monotonic()
    completed_frames = 0
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for plan in plans:
                for segment_number, segment in enumerate(plan.segments):
                    LOGGER.info(
                        "Writing episode %d: %s segment %d (%d frames)",
                        episode_index,
                        plan.scene.name,
                        segment_number,
                        len(segment),
                    )
                    for frame in parallel_frames(
                        pool, plan, segment, args.image_size, args.task, args.prefetch_frames
                    ):
                        dataset.add_frame(frame)
                    dataset.save_episode()
                    plan.report["segments"][segment_number]["episode_index"] = episode_index
                    episode_index += 1
                    completed_frames += len(segment)
                    elapsed = time.monotonic() - started
                    LOGGER.info(
                        "Progress: %d/%d frames, %.1f frames/s, ETA %.1f min",
                        completed_frames,
                        manifest["total_frames"],
                        completed_frames / elapsed,
                        (manifest["total_frames"] - completed_frames) * elapsed / completed_frames / 60,
                    )
    finally:
        dataset.finalize()
    import av

    for key in CAMERA_KEYS.values():
        encoded_frames = 0
        for path in (args.output / "videos" / key).rglob("*.mp4"):
            with av.open(str(path)) as container:
                encoded_frames += container.streams.video[0].frames
        if encoded_frames != manifest["total_frames"]:
            raise RuntimeError(f"{key}: encoded {encoded_frames} / {manifest['total_frames']} frames")
    manifest["status"] = "computing_relative_stats"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    LOGGER.info(
        "Computing joint-relative / gripper-absolute action statistics (chunk_size=%d)", args.chunk_size
    )
    dataset = LeRobotDataset(repo_id=args.repo_id, root=args.output, video_backend="pyav")
    recompute_stats(
        dataset,
        relative_action=True,
        relative_exclude_joints=["gripper"],
        chunk_size=args.chunk_size,
        num_workers=0,
    )
    manifest["status"] = "complete"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="New LeRobot directory; existing paths are refused.")
    parser.add_argument("--repo-id", default="local/template_test_joint_rtc")
    parser.add_argument("--task", help="Actual language instruction for these demonstrations.")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--workers", type=int, default=8, help="Parallel PNG decode/resize workers.")
    parser.add_argument("--prefetch-frames", type=int, default=32, help="Maximum prefetched RGB frame sets.")
    parser.add_argument("--max-camera-skew-ms", type=float, default=100)
    parser.add_argument("--max-lowdim-gap-ms", type=float, default=50)
    parser.add_argument("--max-scenes", type=int, help="Smoke test: select only the first N scenes.")
    parser.add_argument("--max-frames", type=int, help="Smoke test: cap each output episode's length.")
    parser.add_argument("--dry-run", action="store_true", help="Audit all timestamps/labels; write nothing.")
    args = parser.parse_args()
    if not args.dry_run and (args.output is None or not args.task or not args.task.strip()):
        parser.error("conversion requires --output and a nonempty --task")
    if args.fps < 1 or args.chunk_size < 2 or args.image_size < 2 or args.image_size % 2:
        parser.error("fps >= 1, chunk-size >= 2, and an even image-size >= 2 are required")
    if args.max_camera_skew_ms <= 0 or args.max_lowdim_gap_ms <= 0:
        parser.error("timestamp gap limits must be positive")
    if args.workers < 1 or args.prefetch_frames < 1:
        parser.error("workers and prefetch-frames must be positive")
    if args.max_scenes is not None and args.max_scenes < 1:
        parser.error("max-scenes must be positive")
    if args.max_frames is not None and args.max_frames < args.chunk_size:
        parser.error("max-frames must be >= chunk-size for relative statistics")
    return args


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    args.source = args.source.expanduser().resolve(strict=True)
    if args.output is not None:
        args.output = args.output.expanduser().resolve()
        if args.output == args.source or args.source in args.output.parents:
            raise ValueError("Output must be outside the raw source directory.")
        if args.output.exists() and not args.dry_run:
            raise FileExistsError(f"Refusing to overwrite {args.output}")
    layout = read_layout(args.source)
    scenes = sorted(path for path in args.source.glob("scene_*") if path.is_dir())
    if args.max_scenes is not None:
        scenes = scenes[: args.max_scenes]
    if not scenes:
        raise ValueError(f"No scene_* directories under {args.source}")
    plans = []
    for scene in scenes:
        plan = plan_scene(scene, layout, args)
        plans.append(plan)
        LOGGER.info(
            "%s: %d/%d frames -> %d episodes; worst RGB skew %.1f ms before filtering",
            scene.name,
            plan.report["retained_frames"],
            len(plan.times_ms),
            len(plan.segments),
            max(camera["max_skew_ms_before_filter"] for camera in plan.report["cameras"].values()),
        )
    total_frames = sum(plan.report["retained_frames"] for plan in plans)
    total_episodes = sum(len(plan.segments) for plan in plans)
    if not total_frames:
        raise ValueError("No valid segments remain. Inspect the timestamp limits and chunk size.")
    manifest = {
        "source": str(args.source),
        "repo_id": args.repo_id,
        "task": args.task,
        "fps": args.fps,
        "image_size": args.image_size,
        "decode_workers": args.workers,
        "prefetch_frames": args.prefetch_frames,
        "video_encoding": "streaming h264; no intermediate PNG files",
        "relative_stats_chunk_size": args.chunk_size,
        "dimension_names": layout.names,
        "arms": dict(layout.arms),
        "units": {"joint": "radians", "gripper": "meters"},
        "stored_action": "absolute measured joint position at t+1/fps; absolute gripper command at t",
        "joint_label_limitation": "measured-next proxy: original joint commands were not recorded",
        "training_joint_delta": "action[t+k, joint] - observation.state[t, joint], same origin for all k",
        "gripper_command_sampling": "last command with source timestamp <= observation timestamp",
        "image_processing": "nearest timestamp RGB; aspect-preserving square letterbox; no depth",
        "max_camera_skew_ms": args.max_camera_skew_ms,
        "max_lowdim_gap_ms": args.max_lowdim_gap_ms,
        "gap_handling": "split into contiguous episodes; discard segments shorter than chunk_size",
        "total_frames": total_frames,
        "total_episodes": total_episodes,
        "scenes": [plan.report for plan in plans],
    }
    LOGGER.info(
        "TOTAL: %d scenes -> %d episodes, %d frames (%.1f minutes at %d Hz)",
        len(plans),
        total_episodes,
        total_frames,
        total_frames / args.fps / 60,
        args.fps,
    )
    if args.dry_run:
        LOGGER.info("Dry run complete; no output written. Gripper absolute, joint targets absolute on disk.")
        return
    write_dataset(plans, layout, args, manifest)
    LOGGER.info("Done: %s (relative action statistics ready for chunk_size=%d)", args.output, args.chunk_size)


if __name__ == "__main__":
    main()
