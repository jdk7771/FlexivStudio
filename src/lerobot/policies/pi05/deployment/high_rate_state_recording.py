"""Bounded robot-state polling and background JSONL writing, without image capture."""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Thread
from typing import Any

LOGGER = logging.getLogger(__name__)


class StateRecorder:
    def __init__(self, directory: Path, cfg: dict, hz: int) -> None:
        if not 1 <= hz <= 1000:
            raise ValueError("State polling frequency must be between 1 and 1000 Hz")
        self.directory, self.hz = directory, hz
        self._stop, self._done = Event(), Event()
        self._queue: Queue[dict] = Queue(maxsize=2048)
        self._sampler: Thread | None = None
        self._writer: Thread | None = None
        self._previous: dict[str, tuple[int, list[float]]] = {}
        self._closed = False
        self.samples = self.written = self.dropped = self.missed_periods = 0
        self.distinct = {"left": 0, "right": 0}
        self.error: str | None = None
        self.first_sample: float | None = None
        self.last_sample: float | None = None
        self.read_total_s = self.read_max_s = 0.0
        directory.mkdir(parents=True, exist_ok=False)
        metadata = {
            "schema_version": 1,
            "requested_poll_hz": hz,
            "config": cfg,
            "images_recorded": False,
            "rtc": True,
            "queue_capacity": 2048,
            "scope": "After both followers connect, including home alignment, operator wait and policy execution",
            "units": {
                "q": "rad",
                "dq": "rad/s",
                "ddq_estimate": "rad/s^2",
                "tau": "Nm",
                "width": "m",
                "force": "N",
            },
            "timestamp": "robot_timestamp_ns uses each robot's clock; host monotonic bounds each read; clocks may differ",
            "acceleration": "Finite difference of RDK dq using robot timestamps, not a measured acceleration; null for duplicates or invalid/gapped timestamps",
            "frequency": "Polling target only; repeated SDK timestamps are flagged, not new measurements",
            "overflow": "Drop new rows rather than block polling or control",
        }
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        self._stream = (directory / "states.jsonl").open("x", encoding="utf-8", buffering=65536)

    def _arm_sample(self, side: str, robot: Any, gripper: Any) -> dict:
        before = time.monotonic()
        state = robot.states()
        after = time.monotonic()
        stamp = int(state.timestamp[0]) * 1_000_000_000 + int(state.timestamp[1])
        q, dq = list(state.q), list(state.dq)
        if len(q) != 7 or len(dq) != 7 or not all(math.isfinite(x) for x in q + dq):
            raise ValueError(f"{side}: expected seven finite joint positions and velocities")
        previous = self._previous.get(side)
        duplicate = previous is not None and stamp == previous[0]
        acceleration = None
        dt = None if previous is None else (stamp - previous[0]) / 1e9
        if previous is not None and dt is not None and 0 < dt <= 0.1:
            acceleration = [(current - old) / dt for current, old in zip(dq, previous[1], strict=True)]
        if not duplicate:
            self._previous[side] = stamp, dq
            self.distinct[side] += 1
        grip = gripper.states()
        return {
            "robot_timestamp_ns": stamp,
            "host_read_start_s": before,
            "host_read_end_s": after,
            "duplicate_timestamp": duplicate,
            "dq_delta_time_s": dt,
            "q": q,
            "dq": dq,
            "ddq_estimate": acceleration,
            "tau": list(state.tau),
            "tau_ext": list(state.tau_ext),
            "tcp_pose": list(state.tcp_pose),
            "tcp_vel": list(state.tcp_vel),
            "ext_wrench_in_tcp": list(state.ext_wrench_in_tcp),
            "gripper": {
                "width": float(grip.width),
                "force": float(grip.force),
                "is_moving": bool(grip.is_moving),
            },
            "gripper_read_end_s": time.monotonic(),
        }

    def sample(self, robots: dict, grippers: dict) -> dict:
        return {
            "sequence": self.samples,
            "wall_time": time.time(),
            "monotonic_s": time.monotonic(),
            "left": self._arm_sample("left", robots["left"], grippers["left"]),
            "right": self._arm_sample("right", robots["right"], grippers["right"]),
        }

    def submit(self, row: dict) -> None:
        try:
            self._queue.put_nowait(row)
        except Full:
            self.dropped += 1

    def _poll(self, robots: dict, grippers: dict) -> None:
        interval = 1 / self.hz
        deadline = time.monotonic()
        try:
            while not self._stop.is_set():
                started = time.monotonic()
                row = self.sample(robots, grippers)
                elapsed = time.monotonic() - started
                self.read_total_s += elapsed
                self.read_max_s = max(self.read_max_s, elapsed)
                if self.first_sample is None:
                    self.first_sample = started
                self.last_sample = started
                self.samples += 1
                self.submit(row)
                deadline += interval
                now = time.monotonic()
                if deadline < now:
                    skipped = int((now - deadline) / interval) + 1
                    self.missed_periods += skipped
                    deadline += skipped * interval
                self._stop.wait(max(0, deadline - time.monotonic()))
        except Exception as exc:
            self.error = f"State read failed: {type(exc).__name__}: {exc}"
            LOGGER.error("%s; state recording stopped", self.error)
            self._stop.set()
        finally:
            self._done.set()

    def _write(self) -> None:
        try:
            while not self._done.is_set() or not self._queue.empty():
                try:
                    row = self._queue.get(timeout=0.1)
                except Empty:
                    continue
                self._stream.write(json.dumps(row, allow_nan=False) + "\n")
                self.written += 1
                if self.written % self.hz == 0:
                    self._stream.flush()
        except Exception as exc:
            self.error = f"State write failed: {type(exc).__name__}: {exc}"
            LOGGER.error("%s; inference continues without state recording", self.error)
            self._stop.set()
        finally:
            try:
                self._stream.close()
            except OSError as exc:
                self.error = f"State file close failed: {exc}"

    def start(self, robots: dict, grippers: dict) -> None:
        self._writer = Thread(target=self._write, name="robot-state-writer", daemon=True)
        self._sampler = Thread(
            target=self._poll, args=(robots.copy(), grippers.copy()), name="robot-state-sampler", daemon=True
        )
        self._writer.start()
        try:
            self._sampler.start()
        except BaseException:
            self._done.set()
            raise
        LOGGER.info("Robot states: requested %d Hz; no images; %s", self.hz, self.directory)

    def request_stop(self) -> None:
        self._stop.set()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.request_stop()
        if self._sampler is not None and self._sampler.ident is not None:
            self._sampler.join(timeout=1)
            if self._sampler.is_alive():
                self.error = "State reader did not stop within 1 second"
        else:
            self._done.set()
        if self._writer is not None and self._writer.ident is not None:
            self._writer.join(timeout=5)
            if self._writer.is_alive():
                self.error = "State writer drain timed out; file may still be incomplete"
        else:
            self._stream.close()
        duration = (
            0.0
            if self.first_sample is None or self.last_sample is None
            else self.last_sample - self.first_sample
        )
        summary = {
            "requested_hz": self.hz,
            "samples": self.samples,
            "written": self.written,
            "dropped": self.dropped,
            "unwritten": self.samples - self.written - self.dropped,
            "sample_span_s": duration,
            "achieved_poll_hz": (self.samples - 1) / duration if duration > 0 else 0,
            "distinct_robot_timestamps": self.distinct,
            "missed_periods": self.missed_periods,
            "mean_read_ms": 1000 * self.read_total_s / max(1, self.samples),
            "max_read_ms": 1000 * self.read_max_s,
            "error": self.error,
        }
        try:
            (self.directory / "summary.json").write_text(
                json.dumps(summary, indent=2) + "\n", encoding="utf-8"
            )
        except OSError:
            LOGGER.exception("Failed to save state recording summary")
        LOGGER.info("Robot state recording summary: %s", summary)
