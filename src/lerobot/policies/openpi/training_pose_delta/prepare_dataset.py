"""Build an independent TCP dataset from raw HDF5 and the existing video alignment.

Reads source datasets only. Copies small parquet/metadata files and symlinks videos;
never re-encodes video. Writes to a new .incomplete directory, then renames atomically.
"""

import argparse
import hashlib
import json
import logging
import math
from pathlib import Path
import shutil

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from pose_math import ACTION_NAMES, POSE_NAMES, STATE_NAMES, interpolate_tcp, pose_delta, state_features


def statistics(values: np.ndarray) -> dict:
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Statistics require a nonempty finite matrix.")
    result = {"min": values.min(0).tolist(), "max": values.max(0).tolist(),
              "mean": values.mean(0, dtype=np.float64).tolist(),
              "std": values.std(0, dtype=np.float64).tolist(), "count": [len(values)]}
    quantiles = np.quantile(values, [0.01, 0.10, 0.50, 0.90, 0.99], axis=0)
    result.update({name: row.tolist() for name, row in zip(
        ("q01", "q10", "q50", "q90", "q99"), quantiles, strict=True)})
    return result


def episode_split(rows: list[dict], fraction: float) -> tuple[list[int], list[int]]:
    by_task = {}
    for row in sorted(rows, key=lambda row: row["episode_index"]):
        tasks = row["tasks"]
        by_task.setdefault(tasks[0] if tasks else "", []).append(int(row["episode_index"]))
    train, evaluation = [], []
    for ids in by_task.values():
        count = math.ceil(len(ids) * fraction)
        boundary = len(ids) - count
        train.extend(ids[:boundary])
        evaluation.extend(ids[boundary:])
    if not train or not evaluation:
        raise ValueError("Both train and validation splits must be nonempty.")
    return train, evaluation


def action_chunks(actions: np.ndarray, horizon: int) -> np.ndarray:
    """Match the LeRobot reader's repeat-last padding within each episode."""
    if horizon < 1 or not len(actions):
        raise ValueError("Expected positive horizon and a nonempty episode.")
    indices = np.minimum(np.arange(len(actions))[:, None] + np.arange(horizon), len(actions) - 1)
    return actions[indices]


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def prepare(config: dict) -> None:
    source = Path(config["preparation"]["source_root"]).resolve()
    aligned = Path(config["preparation"]["aligned_joint_root"]).resolve()
    output = Path(config["dataset"]["root"]).resolve()
    if output == aligned or output == source or aligned in output.parents or source in output.parents:
        raise ValueError("Pose dataset must be separate from both source datasets.")
    staging = output.with_name(output.name + ".incomplete")
    if output.exists() or staging.exists():
        raise FileExistsError(f"Refusing to overwrite {output} or {staging}.")
    manifest_path = aligned / "meta/conversion_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    info = json.loads((aligned / "meta/info.json").read_text())
    if manifest.get("status") != "complete" or info["features"]["action"]["shape"] != [16]:
        raise ValueError("Expected a complete 16D joint dataset with a conversion manifest.")
    missing = [str(source / scene["scene"] / "lowdim/lowdim.h5") for scene in manifest["scenes"]
               if not (source / scene["scene"] / "lowdim/lowdim.h5").is_file()]
    if missing:
        raise FileNotFoundError("Missing raw TCP recordings:\n" + "\n".join(missing))
    if not (aligned / "videos").is_dir():
        raise FileNotFoundError(aligned / "videos")
    episodes = {}
    for path in sorted((aligned / "meta/episodes").glob("*/*.parquet")):
        for row in pq.read_table(path).to_pylist():
            episodes[int(row["episode_index"])] = row
    train_ids, eval_ids = episode_split(list(episodes.values()), config["dataset"]["eval_split"])
    if set(episodes) != set(range(info["total_episodes"])):
        raise ValueError("Expected contiguous episode indices.")
    total = info["total_frames"]
    states, actions = np.full((total, 16), np.nan, np.float32), np.full((total, 16), np.nan, np.float32)
    step_ms = 1000 / info["fps"]
    reports = []
    seen = set()
    for scene in manifest["scenes"]:
        h5path = source / scene["scene"] / "lowdim/lowdim.h5"
        with h5py.File(h5path, "r") as h5:
            times = np.asarray(h5["timestamp"], dtype=np.float64).reshape(-1)
            if len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) < 0):
                raise ValueError(f"Invalid timestamps in {h5path}")
            keep = np.r_[np.diff(times) != 0, True]
            unique = times[keep]
            signals = {}
            for side in ("left", "right"):
                serial = manifest["arms"][side]
                for prefix, width in (("tcp_pose", 7), ("ee_state", 1), ("ee_command", 1)):
                    key = f"{prefix}_{serial}"
                    value = np.asarray(h5[key], dtype=np.float64)
                    if value.shape != (len(times), width) or not np.isfinite(value).all():
                        raise ValueError(f"Invalid {key} in {h5path}")
                    signals[(side, prefix)] = value[keep]
        for segment in scene["segments"]:
            index = int(segment["episode_index"])
            if index in seen:
                raise ValueError(f"Repeated episode {index}")
            seen.add(index)
            row = episodes[index]
            start, stop = int(row["dataset_from_index"]), int(row["dataset_to_index"])
            count = int(segment["frames"])
            if stop - start != count or row["length"] != count:
                raise ValueError(f"Episode {index}: manifest/metadata length mismatch.")
            query = segment["source_start_ms"] + np.arange(count) * step_ms
            if abs(query[-1] - segment["source_end_ms"]) > 0.01:
                raise ValueError(f"Episode {index}: timestamp grid mismatch.")
            for gap in np.flatnonzero(np.diff(unique) > config["preparation"]["max_lowdim_gap_ms"]):
                if np.any((query < unique[gap + 1]) & (query + step_ms > unique[gap])):
                    raise ValueError(f"Episode {index}: label crosses an excessive lowdim gap.")
            command_indices = np.searchsorted(unique, query, side="right") - 1
            if np.any(command_indices < 0):
                raise ValueError("No past gripper command at observation time.")
            for arm, side in enumerate(("left", "right")):
                offset = arm * 8
                tcp = signals[(side, "tcp_pose")]
                states[start:stop, offset:offset + 7] = interpolate_tcp(unique, tcp, query)
                actions[start:stop, offset:offset + 7] = interpolate_tcp(unique, tcp, query + step_ms)
                states[start:stop, offset + 7] = np.interp(query, unique, signals[(side, "ee_state")][:, 0])
                actions[start:stop, offset + 7] = signals[(side, "ee_command")][command_indices, 0]
        reports.append({"scene": scene["scene"], "h5": str(h5path), "episodes": len(scene["segments"])})
        logging.info("Prepared TCP labels: %s", scene["scene"])
    if seen != set(episodes) or not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise ValueError("Manifest does not cover the entire dataset.")

    aligned_tables = []
    covered = np.zeros(total, dtype=bool)
    max_gripper_error = 0.0
    for path in sorted((aligned / "data").glob("*/*.parquet")):
        table = pq.read_table(path)
        indices = table["index"].to_numpy()
        if (indices < 0).any() or (indices >= total).any() or len(np.unique(indices)) != len(indices) or covered[indices].any():
            raise ValueError("Parquet global indices are duplicated or out of bounds.")
        covered[indices] = True
        frame_indices = table["frame_index"].to_numpy()
        episode_indices = table["episode_index"].to_numpy()
        for episode_id in np.unique(episode_indices):
            mask = episode_indices == episode_id
            row = episodes[int(episode_id)]
            if not np.array_equal(indices[mask] - row["dataset_from_index"], frame_indices[mask]):
                raise ValueError("Parquet indices do not match the alignment manifest.")
        for name, values in (("observation.state", states), ("action", actions)):
            old = np.asarray(table[name].to_pylist(), dtype=np.float32)
            error = np.abs(old[:, [7, 15]] - values[indices][:, [7, 15]])
            max_gripper_error = max(max_gripper_error, float(error.max()))
            # Reconstructing an epoch-ms grid from rounded segment starts can change
            # interpolated state by a few micrometers. Keep the exact aligned gripper
            # labels after a tight 5 um check; commands must still match exactly.
            tolerance = 5e-6 if name == "observation.state" else 1e-7
            if error.max() > tolerance:
                raise ValueError(f"Gripper alignment mismatch in {path}, {name}: max_error={error.max()}")
            values[np.ix_(indices, [7, 15])] = old[:, [7, 15]]
        aligned_tables.append((path, table))
    if not covered.all():
        raise ValueError("Parquet files do not cover every expected frame.")

    # Fit after the same chunk transform used at training time, on training episodes only.
    deltas, train_states = [], []
    horizon = config["policy"]["chunk_size"]
    for index in train_ids:
        row = episodes[index]
        section = slice(int(row["dataset_from_index"]), int(row["dataset_to_index"]))
        current, target = states[section], actions[section]
        deltas.append(pose_delta(current[:, None, :], action_chunks(target, horizon),
                                 config["pose_delta"]["delta_frame"]).reshape(-1, 14))
        train_states.append(state_features(current))
    norm_stats = {
        "pose_delta": config["pose_delta"], "chunk_size": horizon,
        "train_episodes": train_ids, "eval_episodes": eval_ids,
        "state_names": STATE_NAMES, "action_names": ACTION_NAMES,
        "observation.state": statistics(np.concatenate(train_states)),
        "action": statistics(np.concatenate(deltas)),
        "fit": "training episodes only; all anchors and chunk offsets, repeat-last episode padding",
    }
    del deltas, train_states
    output.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    shutil.copytree(aligned / "meta", staging / "meta")
    (staging / "videos").symlink_to(aligned / "videos", target_is_directory=True)
    for key in ("observation.state", "action"):
        info["features"][key] = {"dtype": "float32", "shape": [16], "names": POSE_NAMES}
    info["robot_type"] = "flexiv_dual_arm_tcp"
    write_json(staging / "meta/info.json", info)
    for path, table in aligned_tables:
        indices = table["index"].to_numpy()
        for name, values in (("observation.state", states), ("action", actions)):
            column = table.schema.get_field_index(name)
            table = table.set_column(column, table.schema.field(column),
                                     pa.array(values[indices].tolist(), type=table.schema.field(column).type))
        target = staging / path.relative_to(aligned)
        target.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, target)
    # Generic dataset metadata describes the stored absolute 16D arrays, not model deltas.
    stats = json.loads((staging / "meta/stats.json").read_text())
    stats.update({"observation.state": statistics(states), "action": statistics(actions)})
    write_json(staging / "meta/stats.json", stats)
    for path in sorted((staging / "meta/episodes").glob("*/*.parquet")):
        table = pq.read_table(path)
        rows = table.to_pylist()
        for row in rows:
            section = slice(int(row["dataset_from_index"]), int(row["dataset_to_index"]))
            for name, values in (("observation.state", states), ("action", actions)):
                for key, value in statistics(values[section]).items():
                    row[f"stats/{name}/{key}"] = value
        pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
    write_json(staging / config["pose_delta"]["stats_file"], norm_stats)
    write_json(staging / "meta/conversion_manifest.json", {
        "status": "complete", "source": str(source), "aligned_joint_root": str(aligned),
        "alignment_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "repo_id": config["dataset"]["repo_id"], "total_frames": total,
        "total_episodes": len(episodes), "fps": info["fps"], "arms": manifest["arms"],
        "stored_pose": config["pose_delta"]["stored_pose"], "quaternion_order": "wxyz",
        "position_unit": "m", "reference_frame": "each arm's own robot base",
        "stored_action": "measured TCP pose at t+1/fps; absolute gripper command at t",
        "label_limitation": "measured-next proxy; commanded TCP targets were not recorded",
        "pose_interpolation": "linear xyz and shortest-path quaternion SLERP",
        "gripper_labels": "exact copies from the aligned joint dataset, after raw-source cross-check",
        "max_gripper_reconstruction_error_m": max_gripper_error,
        "video_reuse": "read-only symlink to aligned dataset videos", "scenes": reports,
    })
    staging.rename(output)
    logging.info("Complete: %s (%d frames, %d train / %d eval episodes)",
                 output, total, len(train_ids), len(eval_ids))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    prepare(json.loads(args.config.read_text()))


if __name__ == "__main__":
    main()
