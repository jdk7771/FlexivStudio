#!/usr/bin/env python
"""Open grippers, lift, return above the recorded home, descend, then close."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from deploy_dualarm import read_start_pose, require_compatible_rdk
from flexiv_deploy_contract import read_json, validate_settings
from flexiv_deploy_hardware import FlexivDualArm
from go_home_path import DEFAULT_LIFT_M, TCP_SPEED_M_S, read_home_tcp, return_to_home, validate_lift

LOGGER = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/deploy_config.json"
    )
    parser.add_argument(
        "--lift-m",
        type=float,
        default=DEFAULT_LIFT_M,
        help="Clearance above home TCP Z; keep current height if higher (meters)",
    )
    parser.add_argument(
        "--check", action="store_true", help="Show the recorded first pose without connecting devices"
    )
    args = parser.parse_args()
    validate_lift(args.lift_m)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = read_json(args.config)
    validate_settings(cfg, rtc=False)
    recorded, source = read_start_pose(cfg)
    tcp = read_home_tcp(cfg, source)
    initial = recorded.astype("float64", copy=True)
    initial[[7, 15]] = cfg["hardware"]["gripper_min_m"]
    LOGGER.info("Collection first frame: %s at %.1f ms", source["scene"], source["timestamp_ms"])
    for side, offset in (("left", 0), ("right", 8)):
        joints = ", ".join(f"{initial[offset + index]:.3f}" for index in range(7))
        LOGGER.info("%s joints (rad): [%s]; gripper: %.4f m", side, joints, initial[offset + 7])
    LOGGER.info(
        "Sequence: open grippers, lift to max(current Z, home Z + %.3f m), transfer above home, descend, close",
        args.lift_m,
    )
    LOGGER.info("Recorded home TCP Z: left %.4f m, right %.4f m (RDK WORLD/base)", tcp[0, 2], tcp[1, 2])
    LOGGER.info(
        "Home + clearance TCP Z: left %.4f m, right %.4f m; nominal path speed %.3f m/s",
        tcp[0, 2] + args.lift_m,
        tcp[1, 2] + args.lift_m,
        TCP_SPEED_M_S,
    )
    if args.check:
        LOGGER.info("File check complete. No devices connected.")
        return

    require_compatible_rdk()
    robot = FlexivDualArm(cfg["hardware"], start_pose=cfg["start_pose"])
    try:
        robot.connect(with_cameras=False)
        return_to_home(robot, cfg, recorded, tcp, args.lift_m)
    finally:
        robot.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        LOGGER.info("Stopped by operator")
