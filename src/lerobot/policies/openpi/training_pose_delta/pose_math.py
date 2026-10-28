"""Dual-arm TCP geometry; stored layout per arm is xyz, quaternion wxyz, gripper.

All chunk targets share the current observation as origin. Rotations map TCP axes
into the corresponding arm's base frame. No shared world calibration is assumed.
"""

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


POSE_NAMES = [f"{side}_{name}" for side in ("left", "right")
              for name in ("x", "y", "z", "qw", "qx", "qy", "qz", "gripper")]
ACTION_NAMES = [f"{side}_{name}" for side in ("left", "right")
                for name in ("dx", "dy", "dz", "drx", "dry", "drz", "gripper")]
STATE_NAMES = [f"{side}_{name}" for side in ("left", "right")
               for name in ("x", "y", "z", "r00", "r10", "r20", "r01", "r11", "r21", "gripper")]


def checked_pose(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if value.shape[-1] != 16 or not np.isfinite(value).all():
        raise ValueError("Expected finite dual-arm xyz/quaternion-wxyz/gripper vectors of width 16.")
    for start in (0, 8):
        norm = np.linalg.norm(value[..., start + 3:start + 7], axis=-1)
        if np.any(np.abs(norm - 1) > 0.02):
            raise ValueError("TCP quaternion norm differs from one by more than 0.02.")
    return value


def rotation(quaternion_wxyz: np.ndarray) -> Rotation:
    return Rotation.from_quat(np.asarray(quaternion_wxyz)[..., [1, 2, 3, 0]].reshape(-1, 4))


def state_features(pose: np.ndarray) -> np.ndarray:
    """20D state: xyz + first two rotation matrix columns + absolute gripper, per arm."""
    pose = checked_pose(pose)
    result = []
    for start in (0, 8):
        matrix = rotation(pose[..., start + 3:start + 7]).as_matrix().reshape(*pose.shape[:-1], 3, 3)
        result.extend((pose[..., start:start + 3], matrix[..., :, 0], matrix[..., :, 1],
                       pose[..., start + 7:start + 8]))
    return np.concatenate(result, axis=-1).astype(np.float32)


def pose_delta(current: np.ndarray, targets: np.ndarray, frame: str = "base") -> np.ndarray:
    """Compute 14D deltas with broadcasting; use current[:, None, :] for a batch of chunks."""
    if frame not in ("base", "tcp"):
        raise ValueError("delta_frame must be 'base' or 'tcp'.")
    current, targets = np.broadcast_arrays(checked_pose(current), checked_pose(targets))
    result = []
    for start in (0, 8):
        origin = rotation(current[..., start + 3:start + 7])
        target = rotation(targets[..., start + 3:start + 7])
        translation = (targets[..., start:start + 3] - current[..., start:start + 3]).reshape(-1, 3)
        if frame == "base":
            relative = target * origin.inv()
        else:
            relative = origin.inv() * target
            translation = origin.inv().apply(translation)
        shape = (*targets.shape[:-1], 3)
        result.extend((translation.reshape(shape), relative.as_rotvec().reshape(shape),
                       targets[..., start + 7:start + 8]))
    return np.concatenate(result, axis=-1).astype(np.float32)


def apply_pose_delta(current: np.ndarray, deltas: np.ndarray, frame: str = "base") -> np.ndarray:
    """Inverse geometry for unnormalized predictions; returns absolute xyz/wxyz/gripper."""
    current = checked_pose(current)
    deltas = np.asarray(deltas, dtype=np.float64)
    if deltas.shape[-1] != 14 or not np.isfinite(deltas).all() or frame not in ("base", "tcp"):
        raise ValueError("Expected finite 14D deltas and frame base/tcp.")
    shape = np.broadcast_shapes(current.shape[:-1], deltas.shape[:-1])
    current = np.broadcast_to(current, (*shape, 16))
    deltas = np.broadcast_to(deltas, (*shape, 14))
    result = []
    for arm in (0, 1):
        start, action = arm * 8, arm * 7
        origin = rotation(current[..., start + 3:start + 7])
        relative = Rotation.from_rotvec(deltas[..., action + 3:action + 6].reshape(-1, 3).copy())
        translation = deltas[..., action:action + 3].reshape(-1, 3).copy()
        if frame == "base":
            target = relative * origin
        else:
            target = origin * relative
            translation = origin.apply(translation)
        result.extend((current[..., start:start + 3] + translation.reshape(*shape, 3),
                       target.as_quat()[..., [3, 0, 1, 2]].reshape(*shape, 4),
                       deltas[..., action + 6:action + 7]))
    return np.concatenate(result, axis=-1).astype(np.float32)


def interpolate_tcp(times_ms: np.ndarray, tcp: np.ndarray, query_ms: np.ndarray) -> np.ndarray:
    """Linear xyz interpolation and shortest-path quaternion SLERP, without extrapolation."""
    times_ms, tcp, query_ms = (np.asarray(v, dtype=np.float64) for v in (times_ms, tcp, query_ms))
    if (times_ms.ndim != 1 or tcp.shape != (len(times_ms), 7) or len(times_ms) < 2
            or not np.isfinite(times_ms).all() or not np.isfinite(tcp).all()
            or not np.isfinite(query_ms).all() or np.any(np.diff(times_ms) <= 0)):
        raise ValueError("Invalid TCP samples or non-increasing timestamps.")
    if np.any(np.abs(np.linalg.norm(tcp[:, 3:], axis=-1) - 1) > 0.02):
        raise ValueError("Invalid TCP quaternion norm.")
    relative_times = (times_ms - times_ms[0]) / 1000
    queries = (query_ms - times_ms[0]) / 1000
    if np.any(queries < 0) or np.any(queries > relative_times[-1]):
        raise ValueError("TCP interpolation would extrapolate outside the recording.")
    xyz = np.column_stack([np.interp(queries, relative_times, tcp[:, i]) for i in range(3)])
    quaternions = Slerp(relative_times, rotation(tcp[:, 3:]))(queries).as_quat()[:, [3, 0, 1, 2]]
    return np.column_stack((xyz, quaternions)).astype(np.float32)
