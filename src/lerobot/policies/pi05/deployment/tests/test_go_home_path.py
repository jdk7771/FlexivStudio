"""Raised home geometry, IK preflight and tracking checks; no hardware is used."""

from __future__ import annotations

import json
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
from flexiv_deploy_contract import read_json
from go_home_path import (
    JOINTS,
    align_deployment_start_pose,
    checked_pose,
    execute_home,
    plan_home,
    raised_poses,
    read_home_tcp,
    return_to_home,
    validate_lift,
)


def tcp_from_q(q: np.ndarray) -> np.ndarray:
    return np.r_[q[:3], np.cos(q[3] / 2), 0, 0, np.sin(q[3] / 2)]


class FakeModel:
    def __init__(self, arm: SimpleNamespace):
        pass

    def reachable(self, pose: list[float], seed: list[float], free: bool) -> tuple[bool, list[float]]:
        if free:
            raise AssertionError("Orientation must stay constrained")
        q = np.asarray(seed).copy()
        q[:3] = pose[:3]
        q[3] = 2 * np.arctan2(pose[6], pose[3])
        return True, q.tolist()


class FakeFollowers:
    def __init__(self) -> None:
        self.q = np.array([0.4, -0.1, 0.1, 0, 0, 0, 0, 0.1, 0.4, 0.1, 0.3, 0, 0, 0, 0, 0.1])
        self.sent: list[np.ndarray] = []
        self.robots = {}
        for side, offset in (("left", 0), ("right", 8)):
            self.robots[side] = SimpleNamespace(
                states=lambda offset=offset: SimpleNamespace(
                    tcp_pose=tcp_from_q(self.q[offset : offset + 7])
                ),
                info=lambda: SimpleNamespace(q_min=[-3.0] * 7, q_max=[3.0] * 7),
            )
        self.wait_for_target = MagicMock()
        self.wait_for_grippers_closed = MagicMock(return_value=(0.0, 0.0))

    def read_state(self) -> np.ndarray:
        return self.q.copy()

    def _query_with_retry(self, query, robot, operation):
        return query()

    def check_initial_pose(
        self, target: np.ndarray, *, enforce_distance_limit: bool = True
    ) -> tuple[np.ndarray, float]:
        distance = float(np.max(np.abs(self.q[JOINTS] - target[JOINTS])))
        if enforce_distance_limit and distance > 1.0:
            raise ValueError("Initial pose too far")
        return self.q.copy(), distance

    def send_action(self, target: np.ndarray, *, force_gripper_command: bool = False) -> np.ndarray:
        self.q = target.copy()
        self.sent.append(target.copy())
        return target.copy()


class TestRaisedPath(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = read_json(Path(__file__).resolve().parents[1] / "configs/deploy_config.json")
        self.robot = FakeFollowers()
        self.target = self.robot.q.copy()
        self.target[[0, 2, 8, 10]] = [0.5, 0.2, 0.5, 0.15]
        self.target[[4, 12]] = [0.1, -0.1]
        self.tcp = np.asarray([tcp_from_q(self.target[offset : offset + 7]) for offset in (0, 8)])
        self.enterContext(patch.dict("sys.modules", {"flexivrdk": SimpleNamespace(Model=FakeModel)}))

    def test_lift_then_transfer_then_descend_in_base_z(self) -> None:
        current = np.asarray([self.robot.robots[side].states().tcp_pose for side in ("left", "right")])
        poses = raised_poses(current, self.tcp, 0.1)
        height = np.maximum(current[:, 2], self.tcp[:, 2] + 0.1)
        previous = current
        stages: list[str] = []
        for stage, _, pose in poses:
            if not stages or stages[-1] != stage:
                stages.append(stage)
            if stage == "lift":
                np.testing.assert_allclose(pose[:, :2], current[:, :2])
                np.testing.assert_allclose(pose[:, 3:], current[:, 3:])
                self.assertTrue(np.all(pose[:, 2] >= previous[:, 2] - 1e-12))
            elif stage == "transfer":
                np.testing.assert_allclose(pose[:, 2], height)
            else:
                np.testing.assert_allclose(pose[:, :2], self.tcp[:, :2])
                np.testing.assert_allclose(pose[:, 3:], self.tcp[:, 3:])
                self.assertTrue(np.all(pose[:, 2] <= previous[:, 2] + 1e-12))
            previous = pose
        self.assertEqual(stages, ["lift", "transfer", "descend"])
        np.testing.assert_allclose(poses[-1][2], self.tcp)

    def test_clearance_is_relative_to_home_without_extra_lift_from_current_height(self) -> None:
        target = np.array([[0.5, -0.1, 0.1, 1, 0, 0, 0], [0.5, 0.1, 0.05, 1, 0, 0, 0]])
        for current_z, expected_z in (
            ([0.05, 0.02], [0.20, 0.15]),
            ([0.18, 0.12], [0.20, 0.15]),
            ([0.30, 0.25], [0.30, 0.25]),
        ):
            with self.subTest(current_z=current_z):
                current = target.copy()
                current[:, 0] -= 0.1
                current[:, 2] = current_z
                poses = raised_poses(current, target, 0.1)
                for stage, _, pose in poses:
                    if stage == "lift":
                        self.assertTrue(np.all(pose[:, 2] >= current[:, 2] - 1e-12))
                    elif stage == "transfer":
                        np.testing.assert_allclose(pose[:, 2], expected_z)
                    else:
                        np.testing.assert_allclose(pose[:, :2], target[:, :2])
                np.testing.assert_allclose(poses[-1][2], target)

    def test_antipodal_quaternions_do_not_rotate_and_invalid_lifts_rejected(self) -> None:
        current = self.tcp.copy()
        target = self.tcp.copy()
        target[:, 3:] *= -1
        for _, _, pose in raised_poses(current, target, 0.1):
            np.testing.assert_allclose(np.abs(pose[:, 3:]), np.abs(current[:, 3:]))
        for lift in (0, -0.1, 0.31, float("nan")):
            with self.assertRaises(ValueError):
                validate_lift(lift)
        with self.assertRaises(ValueError):
            checked_pose([0] * 7)

    def test_all_ik_is_planned_before_motion_and_final_posture_matches(self) -> None:
        plan = plan_home(self.robot, self.cfg, self.target, self.tcp, 0.1)
        self.assertFalse(self.robot.sent)
        previous = self.robot.q
        for waypoint in plan.waypoints:
            self.assertLessEqual(np.max(np.abs(waypoint.joints[JOINTS] - previous[JOINTS])), 0.05)
            np.testing.assert_allclose(waypoint.joints[[7, 15]], 0.1)
            previous = waypoint.joints
        np.testing.assert_allclose(plan.waypoints[-1].joints, self.target)

    def test_smooth_ik_with_large_joint_steps_is_refined_without_changing_tcp_path(self) -> None:
        original = FakeModel.reachable
        self.target[14] = 0.6

        def curved(
            model: FakeModel, pose: list[float], seed: list[float], free: bool
        ) -> tuple[bool, list[float]]:
            ok, q = original(model, pose, seed, free)
            if pose[1] > 0:
                q[6] = 60 * (pose[0] - 0.4) ** 2
            return ok, q

        current = np.asarray([self.robot.robots[side].states().tcp_pose for side in ("left", "right")])
        with patch.object(FakeModel, "reachable", curved):
            plan = plan_home(self.robot, self.cfg, self.target, self.tcp, 0.1)
        self.assertGreater(len(plan.waypoints), len(raised_poses(current, self.tcp, 0.1)))
        self.assertFalse(self.robot.sent)
        previous = self.robot.q
        previous_poses = current
        height = np.maximum(current[:, 2], self.tcp[:, 2] + 0.1)
        for waypoint in plan.waypoints:
            self.assertLessEqual(np.max(np.abs(waypoint.joints[JOINTS] - previous[JOINTS])), 0.05)
            self.assertLessEqual(
                np.max(np.linalg.norm(waypoint.poses[:, :3] - previous_poses[:, :3], axis=1)),
                0.005 + 1e-12,
            )
            if waypoint.stage == "lift":
                np.testing.assert_allclose(waypoint.poses[:, :2], current[:, :2])
            elif waypoint.stage == "transfer":
                np.testing.assert_allclose(waypoint.poses[:, 2], height)
            else:
                np.testing.assert_allclose(waypoint.poses[:, :2], self.tcp[:, :2])
            previous, previous_poses = waypoint.joints, waypoint.poses
        np.testing.assert_array_equal(plan.waypoints[-1].joints, self.target)
        np.testing.assert_allclose(plan.waypoints[-1].poses, self.tcp)

    def test_discontinuous_ik_still_fails_before_motion_with_joint_details(self) -> None:
        original = FakeModel.reachable
        self.target[14] = 0.2

        def discontinuous(
            model: FakeModel, pose: list[float], seed: list[float], free: bool
        ) -> tuple[bool, list[float]]:
            ok, q = original(model, pose, seed, free)
            if pose[1] > 0:
                q[6] = 0.0 if pose[0] < 0.45 else 0.2
            return ok, q

        with (
            patch.object(FakeModel, "reachable", discontinuous),
            self.assertRaisesRegex(
                ValueError,
                r"during transfer .*after 8 subdivisions; right A7: .*delta=0\.20000 rad \(limit 0\.05000\)",
            ),
        ):
            plan_home(self.robot, self.cfg, self.target, self.tcp, 0.1)
        self.assertFalse(self.robot.sent)

    def test_unreachable_path_and_ik_branch_jump_fail_before_motion(self) -> None:
        original = FakeModel.reachable

        def unreachable(
            model: FakeModel, pose: list[float], seed: list[float], free: bool
        ) -> tuple[bool, list[float]]:
            return (False, []) if pose[2] > 0.25 else original(model, pose, seed, free)

        with (
            patch.object(FakeModel, "reachable", unreachable),
            self.assertRaisesRegex(ValueError, "unreachable"),
        ):
            plan_home(self.robot, self.cfg, self.target, self.tcp, 0.1)
        self.assertFalse(self.robot.sent)

        def jump(
            model: FakeModel, pose: list[float], seed: list[float], free: bool
        ) -> tuple[bool, list[float]]:
            ok, q = original(model, pose, seed, free)
            if not any(np.allclose(pose, goal) for goal in self.tcp):
                q[6] += 0.1
            return ok, q

        with patch.object(FakeModel, "reachable", jump), self.assertRaisesRegex(ValueError, "branch jump"):
            plan_home(self.robot, self.cfg, self.target, self.tcp, 0.1)
        self.assertFalse(self.robot.sent)

    def test_wrong_tool_tcp_calibration_is_rejected_before_motion(self) -> None:
        wrong = self.tcp.copy()
        wrong[:, 0] += 0.02
        with self.assertRaisesRegex(ValueError, "tool calibration"):
            plan_home(self.robot, self.cfg, self.target, wrong, 0.1)
        self.assertFalse(self.robot.sent)

    def test_execute_tracks_all_stages_and_reaches_exact_joints(self) -> None:
        plan = plan_home(self.robot, self.cfg, self.target, self.tcp, 0.1)
        clock = [0.0]

        def sleep(seconds: float) -> None:
            clock[0] += seconds

        with (
            patch("go_home_path.time.monotonic", side_effect=lambda: clock[0]),
            patch("go_home_path.time.sleep", side_effect=sleep),
        ):
            execute_home(self.robot, self.cfg, self.target, plan)
        np.testing.assert_array_equal(self.robot.q, self.target)
        self.assertEqual(len(self.robot.sent), len(plan.waypoints))
        self.robot.wait_for_target.assert_called_once()

    def test_deployment_raised_return_accepts_1_002_rad_and_closes_after_reaching_home(self) -> None:
        self.target[1] = self.robot.q[1] - 1.002
        self.target[[7, 15]] = 0.05
        recorded = self.target.copy()
        tcp = np.asarray([tcp_from_q(self.target[offset : offset + 7]) for offset in (0, 8)])
        source = {"scene": "scene_0001", "timestamp_ms": 150.0}
        with self.assertRaisesRegex(ValueError, "too far"):
            self.robot.check_initial_pose(self.target)
        clock = [0.0]
        with (
            patch("go_home_path.read_home_tcp", return_value=tcp) as read_tcp,
            patch("go_home_path.time.monotonic", side_effect=lambda: clock[0]),
            patch(
                "go_home_path.time.sleep",
                side_effect=lambda seconds: clock.__setitem__(0, clock[0] + seconds),
            ),
            patch("builtins.input", side_effect=AssertionError("unexpected prompt")),
        ):
            align_deployment_start_pose(self.robot, self.cfg, (self.target, source))
        read_tcp.assert_called_once_with(self.cfg, source)
        np.testing.assert_array_equal(self.target, recorded)
        np.testing.assert_allclose(self.robot.q[JOINTS], recorded[JOINTS])
        np.testing.assert_allclose(self.robot.sent[0][[7, 15]], [0.1, 0.1])
        np.testing.assert_allclose(self.robot.sent[-2][JOINTS], recorded[JOINTS])
        np.testing.assert_allclose(self.robot.sent[-2][[7, 15]], [0.1, 0.1])
        np.testing.assert_allclose(self.robot.sent[-1][[7, 15]], [0.0, 0.0])
        self.robot.wait_for_grippers_closed.assert_called_once_with(tolerance_m=0.001, timeout_s=10.0)

    def test_return_preflight_failure_sends_no_arm_or_gripper_command(self) -> None:
        tcp = self.tcp.copy()
        tcp[:, 0] += 0.02
        with self.assertRaisesRegex(ValueError, "tool calibration"):
            return_to_home(self.robot, self.cfg, self.target, tcp)
        self.assertFalse(self.robot.sent)
        self.robot.wait_for_target.assert_not_called()
        self.robot.wait_for_grippers_closed.assert_not_called()

    def test_deployment_near_home_and_closed_grippers_sends_no_home_commands(self) -> None:
        self.robot.q[[7, 15]] = [0.00072, 0.0]
        recorded = self.robot.read_state()
        recorded[[1, 14]] -= 0.019
        recorded[[7, 15]] = 0.0
        before = self.robot.read_state()
        with (
            patch("go_home_path.read_home_tcp", return_value=self.tcp),
            patch("go_home_path.plan_home") as plan,
        ):
            align_deployment_start_pose(
                self.robot, self.cfg, (recorded, {"scene": "scene_0001", "timestamp_ms": 150.0})
            )
        plan.assert_not_called()
        self.assertFalse(self.robot.sent)
        np.testing.assert_array_equal(self.robot.q, before)
        self.robot.wait_for_grippers_closed.assert_not_called()

    def test_near_home_with_open_grippers_only_closes_at_current_joints(self) -> None:
        recorded = self.robot.read_state()
        recorded[14] -= 0.019
        before = self.robot.read_state()
        with patch("go_home_path.plan_home") as plan:
            return_to_home(self.robot, self.cfg, recorded, self.tcp, skip_if_close=True)
        plan.assert_not_called()
        self.assertEqual(len(self.robot.sent), 1)
        np.testing.assert_array_equal(self.robot.sent[0][JOINTS], before[JOINTS])
        np.testing.assert_array_equal(self.robot.sent[0][[7, 15]], [0.0, 0.0])
        self.robot.wait_for_grippers_closed.assert_called_once_with(tolerance_m=0.001, timeout_s=10.0)

    def test_right_arm_outside_home_tolerance_still_runs_raised_return(self) -> None:
        recorded = self.robot.read_state()
        recorded[14] -= 0.021

        def follow(robot: FakeFollowers, cfg: dict, target: np.ndarray, plan: object) -> None:
            robot.q = target.copy()

        with (
            patch("go_home_path.plan_home") as plan,
            patch("go_home_path.execute_home", side_effect=follow) as execute,
        ):
            return_to_home(self.robot, self.cfg, recorded, self.tcp, skip_if_close=True)
        plan.assert_called_once()
        execute.assert_called_once()
        np.testing.assert_array_equal(self.robot.q[JOINTS], recorded[JOINTS])

    def test_explicit_home_keeps_raised_path_when_already_near_home(self) -> None:
        recorded = self.robot.read_state()
        recorded[14] -= 0.019
        with (
            patch("go_home_path.plan_home") as plan,
            patch("go_home_path.execute_home") as execute,
        ):
            return_to_home(self.robot, self.cfg, recorded, self.tcp)
        plan.assert_called_once()
        execute.assert_called_once()

    def test_timeout_identifies_arm_and_each_unmet_condition_without_advancing(self) -> None:
        for joint, offset, condition in (
            (8, 0.004, "position"),
            (11, 0.04, "orientation"),
            (14, 0.03, "joints"),
        ):
            with self.subTest(condition=condition):
                self.robot = FakeFollowers()
                plan = plan_home(self.robot, self.cfg, self.target, self.tcp, 0.1)
                clock = [0.0]
                send = self.robot.send_action

                def miss(
                    target: np.ndarray,
                    send: Callable[[np.ndarray], np.ndarray] = send,
                    joint: int = joint,
                    offset: float = offset,
                ) -> np.ndarray:
                    applied = send(target)
                    self.robot.q[joint] += offset
                    return applied

                def sleep(seconds: float, clock: list[float] = clock) -> None:
                    clock[0] += seconds

                with (
                    patch.object(self.robot, "send_action", side_effect=miss),
                    patch("go_home_path.time.monotonic", side_effect=lambda clock=clock: clock[0]),
                    patch("go_home_path.time.sleep", side_effect=sleep),
                    self.assertLogs("go_home_path", level="INFO") as logs,
                    self.assertRaisesRegex(TimeoutError, f"5 s waypoint timeout.*right: unmet={condition}"),
                ):
                    execute_home(self.robot, self.cfg, self.target, plan)
                self.assertTrue(any("Waiting:" in line for line in logs.output))
                self.assertIn("left: unmet=none", logs.output[-1])
                self.assertIn("waypoint 1/", logs.output[-1])
                self.assertEqual(len(self.robot.sent), 1)
                self.robot.wait_for_target.assert_not_called()

    def test_changed_start_pose_and_dropping_tcp_stop_the_path(self) -> None:
        plan = plan_home(self.robot, self.cfg, self.target, self.tcp, 0.1)
        self.robot.q[0] += 0.1
        with self.assertRaisesRegex(ValueError, "moved after"):
            execute_home(self.robot, self.cfg, self.target, plan)
        self.assertFalse(self.robot.sent)
        self.robot.q = plan.start_joints.copy()
        send = self.robot.send_action

        def drop(target: np.ndarray) -> np.ndarray:
            applied = send(target)
            self.robot.q[2] -= 0.02
            return applied

        with (
            patch.object(self.robot, "send_action", side_effect=drop),
            self.assertRaisesRegex(RuntimeError, "dropped below"),
        ):
            execute_home(self.robot, self.cfg, self.target, plan)
        self.assertEqual(len(self.robot.sent), 1)

    def test_transfer_dip_is_waited_for_but_does_not_relax_arrival_tolerance(self) -> None:
        for recover in (True, False):
            with self.subTest(recover=recover):
                self._check_transfer_dip(recover)

    def _check_transfer_dip(self, recover: bool) -> None:
        robot = FakeFollowers()
        plan = plan_home(robot, self.cfg, self.target, self.tcp, 0.1)
        transfer_index = next(
            index for index, point in enumerate(plan.waypoints) if point.stage == "transfer"
        )
        transfer = plan.waypoints[transfer_index]
        send = robot.send_action
        clock = [0.0]
        recovering = []

        def dip(target: np.ndarray) -> np.ndarray:
            applied = send(target)
            if np.array_equal(target, transfer.joints):
                robot.q[2] -= 0.004
                recovering.append(target[2])
            return applied

        def sleep(seconds: float) -> None:
            clock[0] += seconds
            if recover and recovering:
                # A 4 mm dip must not count as arrival: wait before accepting this point.
                self.assertEqual(len(robot.sent), transfer_index + 1)
                robot.q[2] = recovering.pop()

        with (
            patch.object(robot, "send_action", side_effect=dip),
            patch("go_home_path.time.monotonic", side_effect=lambda: clock[0]),
            patch("go_home_path.time.sleep", side_effect=sleep),
        ):
            if recover:
                execute_home(robot, self.cfg, self.target, plan)
                np.testing.assert_array_equal(robot.q, self.target)
                robot.wait_for_target.assert_called_once()
            else:
                with self.assertRaisesRegex(TimeoutError, "5 s waypoint timeout.*left: unmet=position"):
                    execute_home(robot, self.cfg, self.target, plan)
                robot.wait_for_target.assert_not_called()
                np.testing.assert_array_equal(robot.sent[-1], transfer.joints)

    def test_height_guard_stops_either_arm_before_or_during_motion_with_diagnostics(self) -> None:
        for side, joint in (("left", 2), ("right", 10)):
            for before_command in (True, False):
                with self.subTest(side=side, before_command=before_command):
                    self._check_height_guard(side, joint, before_command)

    def _check_height_guard(self, side: str, joint: int, before_command: bool) -> None:
        robot = FakeFollowers()
        plan = plan_home(robot, self.cfg, self.target, self.tcp, 0.1)
        transfer = next(point for point in plan.waypoints if point.stage == "transfer")
        send = robot.send_action
        clock = [0.0]
        if before_command:
            robot.q[joint] -= 0.0061

        def drop(target: np.ndarray) -> np.ndarray:
            applied = send(target)
            if np.array_equal(target, transfer.joints):
                robot.q[joint] -= 0.0061
            return applied

        with (
            patch.object(robot, "send_action", side_effect=drop),
            patch("go_home_path.time.monotonic", side_effect=lambda: clock[0]),
            patch(
                "go_home_path.time.sleep",
                side_effect=lambda seconds: clock.__setitem__(0, clock[0] + seconds),
            ),
            self.assertLogs("go_home_path", level="ERROR") as logs,
            self.assertRaisesRegex(RuntimeError, "dropped below") as caught,
        ):
            execute_home(robot, self.cfg, self.target, plan)
        message = str(caught.exception)
        self.assertIn(f"{side}: actual Z=", message)
        self.assertIn("downward deviation=6.10 mm (limit 6.00 mm)", message)
        self.assertIn("lower limit Z=", message)
        self.assertIn("before command" if before_command else "during motion", message)
        self.assertIn(message, logs.output[-1])
        if before_command:
            self.assertFalse(robot.sent)
        else:
            np.testing.assert_array_equal(robot.sent[-1], transfer.joints)
        robot.wait_for_target.assert_not_called()

    def test_recorded_tcp_interpolates_same_timestamp_and_short_quaternion_arc(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "lowdim.h5"
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps({"arms": {"left": "063906", "right": "062106"}, "max_lowdim_gap_ms": 50})
            )
            with h5py.File(path, "w") as lowdim:
                lowdim["timestamp"] = [1000, 1000, 1040]
                for serial in ("063906", "062106"):
                    lowdim[f"tcp_pose_{serial}"] = [
                        [0, 0, 0, 1, 0, 0, 0],
                        [0.4, 0, 0.2, 1, 0, 0, 0],
                        [0.6, 0, 0.4, -1, 0, 0, 0],
                    ]
            self.cfg["start_pose"]["manifest"] = str(manifest)
            poses = read_home_tcp(self.cfg, {"source": str(path), "timestamp_ms": 1020})
            np.testing.assert_allclose(poses, [[0.5, 0, 0.3, 1, 0, 0, 0]] * 2)


if __name__ == "__main__":
    unittest.main()
