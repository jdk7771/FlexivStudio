#!/usr/bin/env python
"""Initialize both follower grippers while holding the arms at their current positions."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from deploy_dualarm import require_compatible_rdk
from flexiv_deploy_contract import read_json, validate_settings
from flexiv_deploy_hardware import FlexivDualArm

LOGGER = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("deploy_config.json"))
    parser.add_argument("--check", action="store_true", help="Check configuration without connecting devices")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = read_json(args.config)
    validate_settings(cfg, rtc=False)
    if args.check:
        LOGGER.info("Configuration check complete. No devices connected.")
        return

    require_compatible_rdk()
    robot = FlexivDualArm(cfg["hardware"], start_pose=cfg["start_pose"])
    try:
        robot.connect(with_cameras=False)
        LOGGER.info("Holding followers at current joint positions; initializing both grippers")
        robot.initialize_grippers()
        LOGGER.info("Gripper initialization requests and settling wait complete")
    finally:
        robot.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        LOGGER.info("Stopped by operator")
