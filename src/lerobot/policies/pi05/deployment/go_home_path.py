"""Plan a raised TCP return path while retaining the configured joint controller."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from flexiv_deploy_contract import read_json

LOGGER = logging.getLogger(__name__)
POSITION_STEP_M = 0.005
ROTATION_STEP_RAD = 0.03
POSITION_TOLERANCE_M = 0.003
ORIENTATION_TOLERANCE_RAD = 0.03
FINAL_IK_MATCH_RAD = 0.002
TCP_SPEED_M_S = 0.075
JOINTS = np.r_[0:7, 8:15]


def validate_lift(lift_m: float) -> None:
    if not math.isfinite(lift_m) or not 0.02 <= lift_m <= 0.30:
        raise ValueError("--lift-m must be within [0.02, 0.30] meters")


def checked_pose(pose: Any) -> np.ndarray:
    result = np.asarray(pose, dtype=np.float64).copy()
    if result.shape != (7,) or not np.isfinite(result).all():
        raise ValueError("TCP pose must contain seven finite values")
    norm = float(np.linalg.norm(result[3:]))
    if not 0.95 <= norm <= 1.05:
        raise ValueError("Invalid TCP quaternion; expected [w, x, y, z]")
    result[3:] /= norm
    return result


def orientation_error(a: np.ndarray, b: np.ndarray) -> float:
    return 2.0 * math.acos(float(np.clip(abs(np.dot(a, b)), 0.0, 1.0)))


def interpolate_pose(a: np.ndarray, b: np.ndarray, fraction: float) -> np.ndarray:
    q0, q1 = a[3:], b[3:].copy()
    dot = float(np.dot(q0, q1))
    if dot < 0:
        q1, dot = -q1, -dot
    if dot > 0.9995:
        quaternion = q0 + fraction * (q1 - q0)
    else:
        angle = math.acos(float(np.clip(dot, 0.0, 1.0)))
        quaternion = (math.sin((1 - fraction) * angle) * q0 + math.sin(fraction * angle) * q1) / math.sin(
            angle
        )
    quaternion /= np.linalg.norm(quaternion)
    return np.r_[a[:3] + fraction * (b[:3] - a[:3]), quaternion]


def read_home_tcp(cfg: dict, source: dict) -> np.ndarray:
    """Read TCP poses at exactly the same retained timestamp as the recorded home joints."""
    manifest = read_json(Path(cfg["start_pose"]["manifest"]))
    timestamp = float(source["timestamp_ms"])
    poses = []
    with h5py.File(source["source"], "r") as lowdim:
        timestamps = np.asarray(lowdim["timestamp"], dtype=np.float64)
        if (
            timestamps.ndim != 1
            or len(timestamps) < 2
            or not np.isfinite(timestamps).all()
            or np.any(np.diff(timestamps) < 0)
        ):
            raise ValueError("Invalid recorded TCP timestamps")
        keep = np.r_[np.diff(timestamps) != 0, True]
        times = timestamps[keep]
        if not times[0] <= timestamp <= times[-1]:
            raise ValueError("Home TCP timestamp is outside the recording")
        right = int(np.searchsorted(times, timestamp))
        if (
            right < len(times)
            and times[right] != timestamp
            and times[right] - times[right - 1] > manifest["max_lowdim_gap_ms"]
        ):
            raise ValueError("Home TCP timestamp crosses a recording gap")
        left = max(0, right - 1)
        for side in ("left", "right"):
            key = f"tcp_pose_{manifest['arms'][side]}"
            values = np.asarray(lowdim[key], dtype=np.float64)
            if values.shape != (len(timestamps), 7):
                raise ValueError(f"Invalid recorded {key} shape")
            values = values[keep]
            a, b = checked_pose(values[left]), checked_pose(values[right])
            fraction = (
                0.0
                if times[right] == times[left]
                else (timestamp - times[left]) / (times[right] - times[left])
            )
            poses.append(interpolate_pose(a, b, float(fraction)))
    return np.asarray(poses)


@dataclass
class Waypoint:
    stage: str
    fraction: float
    poses: np.ndarray
    joints: np.ndarray


@dataclass
class HomePlan:
    start_joints: np.ndarray
    start_poses: np.ndarray
    waypoints: list[Waypoint]


def raised_poses(
    current: np.ndarray, target: np.ndarray, lift_m: float
) -> list[tuple[str, float, np.ndarray]]:
    validate_lift(lift_m)
    current = np.asarray([checked_pose(pose) for pose in current])
    target = np.asarray([checked_pose(pose) for pose in target])
    if current.shape != (2, 7) or target.shape != (2, 7):
        raise ValueError("Expected one TCP pose per follower")
    lifted, overhead = current.copy(), target.copy()
    # Clearance is measured from home. Never lower an already higher arm before transfer.
    heights = np.maximum(current[:, 2], target[:, 2] + lift_m)
    lifted[:, 2] = overhead[:, 2] = heights
    result = []
    for stage, start, end in (
        ("lift", current, lifted),
        ("transfer", lifted, overhead),
        ("descend", overhead, target),
    ):
        count = max(
            1, math.ceil(float(np.max(np.linalg.norm(end[:, :3] - start[:, :3], axis=1))) / POSITION_STEP_M)
        )
        count = max(
            count,
            math.ceil(
                max(orientation_error(a[3:], b[3:]) for a, b in zip(start, end, strict=True))
                / ROTATION_STEP_RAD
            ),
        )
        for index in range(1, count + 1):
            fraction = index / count
            result.append(
                (
                    stage,
                    fraction,
                    np.asarray([interpolate_pose(a, b, fraction) for a, b in zip(start, end, strict=True)]),
                )
            )
    return result


def current_tcp(robot: Any) -> np.ndarray:
    return np.asarray([checked_pose(robot.robots[side].states().tcp_pose) for side in ("left", "right")])


def plan_home(
    robot: Any,
    cfg: dict,
    target: np.ndarray,
    tcp: np.ndarray,
    lift_m: float,
    *,
    enforce_start_distance: bool = True,
) -> HomePlan:
    """Preflight all IK solutions for both arms before any return motion or gripper opening."""
    import flexivrdk

    start, _ = robot.check_initial_pose(target, enforce_distance_limit=enforce_start_distance)
    current = current_tcp(robot)
    models = [flexivrdk.Model(robot.robots[side]) for side in ("left", "right")]
    poses = raised_poses(current, tcp, lift_m)
    infos = [robot.robots[side].info() for side in ("left", "right")]

    def solve(index: int, pose: np.ndarray, seed: np.ndarray) -> tuple[bool, list[float]]:
        return robot._query_with_retry(
            lambda: models[index].reachable(pose.tolist(), seed.tolist(), False),
            robot.robots[("left", "right")[index]],
            f"{('left', 'right')[index]} home IK reachability",
        )

    # Validate that recorded TCP/tool calibration still corresponds to the recorded home joints.
    for index, offset in enumerate((0, 8)):
        ok, solution = solve(index, tcp[index], target[offset : offset + 7])
        solution = np.asarray(solution, dtype=np.float64)
        if (
            not ok
            or solution.shape != (7,)
            or not np.isfinite(solution).all()
            or np.max(np.abs(solution - target[offset : offset + 7])) > FINAL_IK_MATCH_RAD
        ):
            raise ValueError(
                f"{('left', 'right')[index]} home TCP does not match recorded joints/tool calibration; no return motion sent"
            )
    previous = start.copy()
    descend_seed = start.copy()
    waypoints = []
    for stage, fraction, desired in poses:
        waypoint = target.copy()
        for index, offset in enumerate((0, 8)):
            seed = previous[offset : offset + 7]
            if stage == "descend":
                seed = (1 - fraction) * descend_seed[offset : offset + 7] + fraction * target[
                    offset : offset + 7
                ]
            ok, solution = solve(index, desired[index], seed)
            q = np.asarray(solution, dtype=np.float64)
            if not ok or q.shape != (7,) or not np.isfinite(q).all():
                raise ValueError(
                    f"{('left', 'right')[index]} {stage} waypoint is unreachable; no return motion sent"
                )
            if np.any(q < np.asarray(infos[index].q_min)) or np.any(q > np.asarray(infos[index].q_max)):
                raise ValueError("Home IK exceeds joint limits; no return motion sent")
            if stage == "descend" and fraction == 1.0:
                if np.max(np.abs(q - target[offset : offset + 7])) > FINAL_IK_MATCH_RAD:
                    raise ValueError(
                        "Home IK final posture differs from recorded joints; no return motion sent"
                    )
                q = target[offset : offset + 7].copy()
            if np.max(np.abs(q - previous[offset : offset + 7])) > cfg["start_pose"]["waypoint_step_rad"]:
                raise ValueError(
                    f"Home IK branch jump or excessive joint step during {stage}; no return motion sent"
                )
            waypoint[offset : offset + 7] = q
        waypoints.append(Waypoint(stage, fraction, desired, waypoint))
        previous = waypoint
        if stage != "descend":
            descend_seed = previous.copy()
    if np.max(np.abs(robot.read_state()[JOINTS] - start[JOINTS])) > cfg["start_pose"]["joint_tolerance_rad"]:
        raise ValueError("Followers moved during home planning; restart to replan")
    LOGGER.info(
        "Home path preflight passed: %d waypoints; left/right clearance Z %.3f / %.3f m",
        len(waypoints),
        np.max([p[2][:, 2] for p in poses], axis=0)[0],
        np.max([p[2][:, 2] for p in poses], axis=0)[1],
    )
    return HomePlan(start.copy(), current, waypoints)


def execute_home(robot: Any, cfg: dict, target: np.ndarray, plan: HomePlan) -> None:
    if (
        np.max(np.abs(robot.read_state()[JOINTS] - plan.start_joints[JOINTS]))
        > cfg["start_pose"]["joint_tolerance_rad"]
    ):
        raise ValueError("Followers moved after home planning; restart to replan")
    deadline = time.monotonic() + cfg["start_pose"]["timeout_s"]
    previous_poses = plan.start_poses
    stage = None
    for point_index, point in enumerate(plan.waypoints, start=1):
        if point.stage != stage:
            stage = point.stage
            LOGGER.info("Home stage: %s", stage)
        started = time.monotonic()
        floor = np.minimum(previous_poses[:, 2], point.poses[:, 2]) - POSITION_TOLERANCE_M
        if np.any(current_tcp(robot)[:, 2] < floor):
            raise RuntimeError(f"Home {stage}: TCP below the planned height corridor; stopping")
        robot.send_action(point.joints)
        next_report = started + 1.0
        while True:
            measured = robot.read_state()
            actual = current_tcp(robot)
            if np.any(actual[:, 2] < floor):
                raise RuntimeError(f"Home {stage}: TCP dropped below the planned height corridor; stopping")
            position_errors = np.linalg.norm(actual[:, :3] - point.poses[:, :3], axis=1)
            rotation_errors = np.asarray(
                [orientation_error(a[3:], b[3:]) for a, b in zip(actual, point.poses, strict=True)]
            )
            joint_errors = np.abs(measured[JOINTS] - point.joints[JOINTS]).reshape(2, 7)
            if (
                np.max(position_errors) <= POSITION_TOLERANCE_M
                and np.max(rotation_errors) <= ORIENTATION_TOLERANCE_RAD
                and np.max(joint_errors) <= cfg["start_pose"]["joint_tolerance_rad"]
            ):
                break
            now = time.monotonic()
            timed_out = now >= min(deadline, started + 5.0)
            if now >= next_report or timed_out:
                details = []
                for index, side in enumerate(("left", "right")):
                    worst_joint = int(np.argmax(joint_errors[index]))
                    unmet = []
                    if position_errors[index] > POSITION_TOLERANCE_M:
                        unmet.append("position")
                    if rotation_errors[index] > ORIENTATION_TOLERANCE_RAD:
                        unmet.append("orientation")
                    if joint_errors[index, worst_joint] > cfg["start_pose"]["joint_tolerance_rad"]:
                        unmet.append("joints")
                    details.append(
                        f"{side}: unmet={','.join(unmet) or 'none'}, "
                        f"TCP={position_errors[index] * 1000:.2f} mm "
                        f"(limit {POSITION_TOLERANCE_M * 1000:.2f}), "
                        f"orientation={rotation_errors[index]:.4f} rad "
                        f"(limit {ORIENTATION_TOLERANCE_RAD:.4f}), "
                        f"A{worst_joint + 1} error={joint_errors[index, worst_joint]:.4f} rad "
                        f"(limit {cfg['start_pose']['joint_tolerance_rad']:.4f}), "
                        f"TCP xyz actual={actual[index, :3].round(5).tolist()} "
                        f"target={point.poses[index, :3].round(5).tolist()} m"
                    )
                message = (
                    f"Home {stage} waypoint {point_index}/{len(plan.waypoints)} "
                    f"({point.fraction:.1%} of stage), elapsed {now - started:.2f} s: " + "; ".join(details)
                )
                if timed_out:
                    reason = "overall return timeout" if now >= deadline else "5 s waypoint timeout"
                    LOGGER.error("%s; %s", reason, message)
                    raise TimeoutError(f"{reason}; {message}")
                LOGGER.info("Waiting: %s", message)
                next_report = now + 1.0
            time.sleep(0.02)
        interval = max(
            float(np.max(np.linalg.norm(point.poses[:, :3] - previous_poses[:, :3], axis=1))) / TCP_SPEED_M_S,
            0.02,
        )
        remaining = interval - (time.monotonic() - started)
        if remaining > 0:
            time.sleep(remaining)
        if time.monotonic() >= deadline:
            raise TimeoutError("Raised home path exceeded the configured return timeout")
        previous_poses = point.poses
    robot.wait_for_target(
        target, cfg["action_timeout_s"], grippers=False, operation="Raised home final joints"
    )
