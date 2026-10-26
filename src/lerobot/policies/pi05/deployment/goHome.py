#!/usr/bin/env python
"""Open both grippers, return to the first collection pose, then close them."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from deploy_dualarm import align_start_pose, read_start_pose, require_compatible_rdk
from flexiv_deploy_contract import read_json, validate_settings
from flexiv_deploy_hardware import FlexivDualArm

LOGGER = logging.getLogger(__name__)
CLOSED_GRIPPER_TOLERANCE_M = 0.001
GRIPPER_MOVE_TIMEOUT_S = 10.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("deploy_config.json"))
    parser.add_argument(
        "--check", action="store_true", help="Show the recorded first pose without connecting devices"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = read_json(args.config)
    validate_settings(cfg)
    recorded, source = read_start_pose(cfg)
    initial = recorded.astype("float64", copy=True)
    initial[[7, 15]] = cfg["hardware"]["gripper_min_m"]
    LOGGER.info("Collection first frame: %s at %.1f ms", source["scene"], source["timestamp_ms"])
    for side, offset in (("left", 0), ("right", 8)):
        joints = ", ".join(f"{initial[offset + index]:.3f}" for index in range(7))
        LOGGER.info("%s joints (rad): [%s]; gripper: %.4f m", side, joints, initial[offset + 7])
    LOGGER.info(
        "Sequence: open both grippers to %.4f m, return arms, then close", cfg["hardware"]["gripper_max_m"]
    )
    if args.check:
        LOGGER.info("File check complete. No devices connected.")
        return

    require_compatible_rdk()
    robot = FlexivDualArm(cfg["hardware"], start_pose=cfg["start_pose"])
    try:
        robot.connect(with_cameras=False)
        # Check the complete return pose before issuing any gripper command.
        measured, _ = robot.check_initial_pose(initial)
        opening = measured.copy()
        opening[[7, 15]] = cfg["hardware"]["gripper_max_m"]
        applied = robot.send_action(opening, force_gripper_command=True)
        robot.wait_for_target(applied, GRIPPER_MOVE_TIMEOUT_S, operation="Opening both grippers")
        LOGGER.info("Both grippers opened; returning arms to the first joint pose")
        open_pose = initial.copy()
        open_pose[[7, 15]] = cfg["hardware"]["gripper_max_m"]
        align_start_pose(robot, (open_pose, source))
        closing = robot.read_state().copy()
        closing[[7, 15]] = initial[[7, 15]]
        robot.send_action(closing, force_gripper_command=True)
        LOGGER.info("Arms reached the first joint pose; closing both grippers")
        left_width, right_width = robot.wait_for_grippers_closed(
            tolerance_m=CLOSED_GRIPPER_TOLERANCE_M, timeout_s=GRIPPER_MOVE_TIMEOUT_S
        )
        LOGGER.info(
            "Both followers reached the first joint pose; RDK gripper widths: left %.4f m, right %.4f m",
            left_width,
            right_width,
        )
    finally:
        robot.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        LOGGER.info("Stopped by operator")
