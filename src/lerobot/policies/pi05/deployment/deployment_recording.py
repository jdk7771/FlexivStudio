"""Bounded, optional RGB/state recording; the writer process never owns hardware or CUDA."""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import time
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from queue import Empty, Full
from typing import Any
from uuid import uuid4

import numpy as np
from flexiv_deploy_contract import ACTION_NAMES, CAMERA_NAMES
from PIL import Image

LOGGER = logging.getLogger(__name__)
SLOTS = 8
FRAME_SHAPE = (len(CAMERA_NAMES), 224, 224, 3)


def add_recording_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--record",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Record policy-input RGB, measured joints/grippers and applied targets in a background process",
    )
    parser.add_argument("--record-dir", type=Path, help="Recording parent directory (one new folder per run)")


def configure_recording(cfg: dict, args: argparse.Namespace) -> None:
    cfg["record"] = args.record
    cfg["record_dir"] = str(
        (args.record_dir or Path(cfg["telemetry"]).parent / "recordings").expanduser().resolve()
    )


def _write_record(root: Path, stream: Any, buffers: np.ndarray, slot: int, payload: dict) -> None:
    paths = {}
    for index, name in enumerate(CAMERA_NAMES):
        relative = f"images/{name}/{payload['sequence']:08d}.jpg"
        Image.fromarray(buffers[slot, index]).save(root / relative, quality=90)
        paths[name] = relative
    # Only index a sample after all three JPEG files have been written successfully.
    stream.write(json.dumps({**payload, "images": paths}, allow_nan=False) + "\n")


def _writer(root: str, telemetry: str, shared: Any, jobs: Any, released: Any, status: Any) -> None:
    """Only small metadata crosses the queue; images live in fixed shared-memory slots."""
    written = 0
    try:
        os.nice(10)
        path = Path(root)
        buffers = np.frombuffer(shared, dtype=np.uint8).reshape((SLOTS, *FRAME_SHAPE))
        for name in CAMERA_NAMES:
            (path / "images" / name).mkdir(parents=True)
        with (
            (path / "frames.jsonl").open("x", encoding="utf-8", buffering=1) as frames,
            Path(telemetry).open("a", encoding="utf-8", buffering=1) as log,
        ):
            status.send({"ready": True})
            while True:
                job = jobs.get()
                if job is None:
                    break
                slot, payload = job
                _write_record(path, frames, buffers, slot, payload)
                log.write(json.dumps(payload["telemetry"], allow_nan=False) + "\n")
                written += 1
                released.put(slot)
        status.send({"written": written, "error": None})
    except Exception as exc:
        status.send({"written": written, "error": f"{type(exc).__name__}: {exc}"})
    finally:
        status.close()


class DeploymentLog:
    """One producer, with nonblocking submission and bounded image memory.

    Construct before connecting devices, submit only during policy execution, and
    close AFTER stopping robots/inference. With recording disabled, retain the
    existing JSONL telemetry behavior and do not create a worker or image buffer.
    """

    def __init__(self, cfg: dict, run_mode: str, *, rtc: bool) -> None:
        self.enabled = bool(cfg.get("record", False))
        self.root: Path | None = None
        self._stream: Any = None
        self._process: Any = None
        self._closed = False
        self._warned = False
        self.attempted = self.submitted = self.dropped = 0
        self.submit_total_ms = self.submit_max_ms = 0.0
        self._status: dict = {}
        telemetry = Path(cfg["telemetry"])
        telemetry.parent.mkdir(parents=True, exist_ok=True)
        if not self.enabled:
            self._stream = telemetry.open("a", encoding="utf-8", buffering=1)
            return
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        parent = Path(cfg.get("record_dir", telemetry.parent / "recordings"))
        self.root = parent / f"{stamp}_{'rtc' if rtc else 'no_rtc'}_{uuid4().hex[:8]}"
        self.root.mkdir(parents=True, exist_ok=False)
        metadata = {
            "schema_version": 1,
            "config": cfg,
            "rtc": rtc,
            "run_mode": run_mode,
            "action_names": ACTION_NAMES,
            "units": {"joint": "rad", "gripper": "m"},
            "image_format": "RGB JPEG quality=90, 224x224 letterboxed policy inputs",
            "clock": "wall_time: Unix seconds; observation timestamps: host monotonic seconds",
            "scope": "Policy control ticks only; RTC observations are not necessarily inference request frames",
            "overflow": "Drop the entire new sample, including its telemetry; never wait for the writer",
            "queue_slots": SLOTS,
        }
        (self.root / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        ctx = mp.get_context("spawn")  # Never fork the loaded CUDA model or RDK state.
        self._shared = ctx.RawArray("B", SLOTS * int(np.prod(FRAME_SHAPE)))
        self._buffers = np.frombuffer(self._shared, dtype=np.uint8).reshape((SLOTS, *FRAME_SHAPE))
        self._free = list(range(SLOTS))
        self._jobs = ctx.Queue(maxsize=SLOTS)
        self._released = ctx.Queue(maxsize=SLOTS)
        self._reader, sender = ctx.Pipe(duplex=False)
        self._process = ctx.Process(
            target=_writer,
            args=(str(self.root), str(telemetry), self._shared, self._jobs, self._released, sender),
            name="deployment-recorder",
            daemon=True,
        )
        try:
            self._process.start()
            sender.close()
            if not self._reader.poll(10):
                raise TimeoutError("Recording writer did not become ready before device connection")
            ready = self._reader.recv()
            if not ready.get("ready"):
                raise RuntimeError(f"Cannot start recording: {ready.get('error', ready)}")
        except BaseException:
            sender.close()
            self.close()
            raise
        LOGGER.info("Recording enabled: %s (three 224x224 RGB views; bounded background writer)", self.root)

    def record(self, telemetry: dict, observation: dict, capture_time: float | None) -> bool:
        if self._closed:
            return False
        if not self.enabled:
            self._stream.write(json.dumps(telemetry) + "\n")
            return True
        started = time.perf_counter()
        sequence = self.attempted
        self.attempted += 1
        try:
            if not self._process.is_alive():
                self.dropped += 1
                if not self._warned:
                    LOGGER.warning("Recording writer stopped; continuing deployment without recording")
                    self._warned = True
                return False
            while True:
                try:
                    self._free.append(self._released.get_nowait())
                except Empty:
                    break
            if not self._free:
                self.dropped += 1
                return False
            slot = self._free.pop()
            try:
                for index, name in enumerate(CAMERA_NAMES):
                    rgb = observation[name]
                    if rgb.shape != FRAME_SHAPE[1:] or rgb.dtype != np.uint8:
                        raise ValueError(f"Recording requires 224x224 RGB uint8: {name}")
                    np.copyto(self._buffers[slot, index], rgb)
                payload = {
                    "sequence": sequence,
                    "telemetry": telemetry,
                    "observation_time_monotonic_s": capture_time,
                    "capture": observation.get("_capture_metadata", {}),
                }
                self._jobs.put_nowait((slot, payload))
            except Full:
                self._free.append(slot)
                self.dropped += 1
                return False
            except Exception as exc:
                self._free.append(slot)
                self.dropped += 1
                if not self._warned:
                    LOGGER.warning("Recording sample rejected: %s; continuing deployment", exc)
                    self._warned = True
                return False
            self.submitted += 1
            return True
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000
            self.submit_total_ms += elapsed_ms
            self.submit_max_ms = max(self.submit_max_ms, elapsed_ms)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._stream is not None:
            self._stream.close()
        if self._process is None:
            return
        # This bounded drain happens only after robot shutdown, never inside a control tick.
        if self._process.pid is not None:
            if self._process.is_alive():
                with suppress(Full):
                    self._jobs.put(None, timeout=1)
                self._process.join(timeout=5)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=1)
                self._status["error"] = "Writer drain timed out; recording is incomplete"
            else:
                try:
                    while self._reader.poll():
                        self._status.update(self._reader.recv())
                except EOFError:
                    pass
            if "written" not in self._status:
                self._status.setdefault(
                    "error", f"Writer exited without a summary (exitcode={self._process.exitcode})"
                )
        self._reader.close()
        for queue in (self._jobs, self._released):
            queue.cancel_join_thread()
            queue.close()
        summary = {
            **self._status,
            "attempted": self.attempted,
            "submitted": self.submitted,
            "dropped_at_submit": self.dropped,
            "unwritten_submissions": self.submitted - self._status.get("written", 0),
            "submit_mean_ms": self.submit_total_ms / max(1, self.attempted),
            "submit_max_ms": self.submit_max_ms,
        }
        try:
            if self.root is not None:
                (self.root / "summary.json").write_text(
                    json.dumps(summary, indent=2) + "\n", encoding="utf-8"
                )
        except OSError:
            LOGGER.exception("Cannot save recording summary")
        LOGGER.info("Recording summary: %s", summary)
