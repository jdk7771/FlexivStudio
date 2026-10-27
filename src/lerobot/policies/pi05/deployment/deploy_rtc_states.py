"""Separate RTC entry point with high-rate robot states and no image recording."""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import deploy_dualarm
import flexiv_deploy_hardware
from high_rate_state_recording import StateRecorder


def recording_robot_class(cfg: dict, hz: int, parent: Path) -> type:
    class RecordingRobot(flexiv_deploy_hardware.FlexivDualArm):
        recorder: StateRecorder | None = None

        def connect(self, *, with_cameras: bool = True) -> None:
            super().connect(with_cameras=with_cameras)
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            directory = parent / f"{stamp}_rtc_states_{uuid4().hex[:8]}"
            self.recorder = StateRecorder(directory, cfg, hz)
            self.recorder.start(self.robots, self.grippers)

        def close(self) -> None:
            if self.recorder is not None:
                self.recorder.request_stop()
            try:
                # Stop motion before joining sampler/writer threads.
                super().close()
            finally:
                if self.recorder is not None:
                    self.recorder.close()

    return RecordingRobot


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--state-hz", type=int, default=200)
    parser.add_argument("--state-dir", type=Path)
    args, remaining = parser.parse_known_args()
    if not 1 <= args.state_hz <= 1000:
        parser.error("--state-hz must be between 1 and 1000")
    if any(value == "--record" or value.startswith("--record=") for value in remaining):
        parser.error("This entry records states only; do not pass --record")
    if "--help" in remaining or "-h" in remaining:
        print(
            "RTC states-only entry: --state-hz 200 (1..1000), --state-dir PATH. Other options below:",
            flush=True,
        )
    original_execute = deploy_dualarm.execute

    def execute(cfg: dict, policy: Any, pre: Any, post: Any, run_mode: str, start_pose: Any) -> None:
        cfg = {**cfg, "record": False, "state_record_hz": args.state_hz, "run_mode": run_mode}
        parent = (args.state_dir or Path(cfg["telemetry"]).parent / "robot_states").expanduser().resolve()
        original_robot = flexiv_deploy_hardware.FlexivDualArm
        flexiv_deploy_hardware.FlexivDualArm = recording_robot_class(cfg, args.state_hz, parent)
        try:
            original_execute(cfg, policy, pre, post, run_mode, start_pose)
        finally:
            flexiv_deploy_hardware.FlexivDualArm = original_robot

    original_argv = sys.argv
    deploy_dualarm.execute = execute
    sys.argv = [original_argv[0], *remaining, "--no-record"]
    try:
        deploy_dualarm.main()
    finally:
        sys.argv = original_argv
        deploy_dualarm.execute = original_execute


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Stopped by operator")
