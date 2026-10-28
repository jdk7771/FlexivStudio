"""Single inference worker with committed RTC prefix and bounded action lifetime."""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CompletedPrediction:
    actions: np.ndarray
    completed_at: float


class ChunkEngine:
    def __init__(self, policy: Any, cfg: dict, *, executor: Any = None, clock: Any = time.monotonic) -> None:
        self.policy, self.cfg, self.clock = policy, cfg, clock
        self.executor = (
            executor
            if executor is not None
            else ThreadPoolExecutor(max_workers=1, thread_name_prefix="openpi")
        )
        self.queue: deque[np.ndarray] = deque()
        self.future: Future | None = None
        self.prefix: np.ndarray = np.empty((0, 16), dtype=np.float32)
        self.consumed = 0
        self.queue_capture = 0.0
        self.request_capture = 0.0
        self.request_started = 0.0
        self.last_inference_ms = 0.0
        self.last_prefix_length = 0
        self.inferences = 0
        self.turnarounds: deque[float] = deque(maxlen=100)
        self.initial_turnaround = 0.0
        self.completed = 0
        self.discarded = 0
        self.consecutive_discards = 0
        self.has_executed = False
        self.last_trimmed = 0

    @property
    def turnaround(self) -> float:
        """Match LeRobot: recent maximum, with cold-start fallback until a warm result exists."""
        return max(self.turnarounds, default=self.initial_turnaround)

    @property
    def refill_seconds(self) -> float:
        return max(self.cfg["rtc_queue_threshold"] / self.cfg["fps"], self.turnaround + 1 / self.cfg["fps"])

    def prefix_length(self) -> int:
        if not self.cfg["rtc"] or not self.queue:
            return 0
        delay = math.ceil(self.turnaround / (1 / self.cfg["fps"])) if self.turnaround else 0
        maximum = self.cfg["rtc_prefix_steps"]
        return min(delay or maximum, maximum, len(self.queue))

    def _check_deadline(self, capture: float, started: float) -> None:
        now = self.clock()
        if now - capture > self.cfg["max_observation_age_s"] or now - started > self.cfg["action_timeout_s"]:
            raise TimeoutError("OpenPI observation/inference expired; no stale action may be dispatched")

    def _predict(self, observation: dict, prefix: np.ndarray) -> CompletedPrediction:
        actions = self.policy.predict(observation, prefix)
        # GPU output is already copied to CPU by predict. Exclude waiting for the next 15 Hz tick.
        return CompletedPrediction(actions, self.clock())

    def _accept(self) -> None:
        if self.future is None:
            return
        self._check_deadline(self.request_capture, self.request_started)
        if not self.future.done():
            return
        result = self.future.result()
        completed_at = result.completed_at if isinstance(result, CompletedPrediction) else self.clock()
        actions = np.asarray(
            result.actions if isinstance(result, CompletedPrediction) else result, dtype=np.float32
        )
        self.future = None
        if actions.shape != (self.cfg["chunk_size"], 16) or not np.isfinite(actions).all():
            raise ValueError("Expected a finite (chunk_size,16) action chunk")
        if not np.allclose(actions[: len(self.prefix)], self.prefix, atol=1e-6, rtol=0):
            raise ValueError("RTC inference changed its committed absolute prefix")
        elapsed = completed_at - self.request_started
        self.completed += 1
        if self.completed == 1:
            self.initial_turnaround = elapsed
        else:
            self.turnarounds.append(elapsed)
        self.last_inference_ms = elapsed * 1000
        measured = math.ceil(elapsed / (1 / self.cfg["fps"]))
        if self.cfg["rtc"] and len(self.prefix) and max(measured, self.consumed) > len(self.prefix):
            # Keep playing the valid old queue; never splice an under-conditioned successor.
            self.discarded += 1
            self.consecutive_discards += 1
            LOGGER.warning(
                "Discarding RTC result: prefix=%d elapsed_steps=%d dispatched=%d (consecutive=%d)",
                len(self.prefix),
                measured,
                self.consumed,
                self.consecutive_discards,
            )
            if self.consecutive_discards >= 5:
                raise RuntimeError(
                    "Trained RTC inference repeatedly exceeded the conditioned overlap (5 results)"
                )
            return
        self.consecutive_discards = 0
        trim = min(measured, self.consumed) if self.cfg["rtc"] else 0
        limit = len(actions) if self.cfg["rtc"] else self.cfg["n_action_steps"]
        self.queue = deque(a.copy() for a in actions[trim:limit])
        self.queue_capture = self.request_capture
        self.last_trimmed = trim
        self.inferences += 1

    def tick(self, observation: dict, capture_time: float) -> np.ndarray | None:
        self._accept()
        needs = (
            len(self.queue) / self.cfg["fps"] <= self.refill_seconds + 1e-9
            if self.cfg["rtc"]
            else not self.queue
        )
        if self.future is None and needs:
            self.request_started = self.clock()
            self.request_capture = capture_time
            self._check_deadline(capture_time, self.request_started)
            count = self.prefix_length()
            self.prefix = np.asarray(list(self.queue)[:count], dtype=np.float32).reshape(count, 16)
            self.last_prefix_length = count
            self.consumed = 0
            # Camera snapshots may be reused by the caller; own all inference inputs.
            snapshot = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in observation.items()}
            self.future = self.executor.submit(self._predict, snapshot, self.prefix.copy())
        if not self.queue:
            if self.cfg["rtc"] and self.has_executed:
                raise RuntimeError("Active RTC motion buffer exhausted")
            return None  # Initial inference or ordinary non-RTC chunk boundary.
        if self.clock() - self.queue_capture > self.cfg["max_observation_age_s"]:
            raise TimeoutError("Action chunk observation expired")
        action = self.queue.popleft()
        self.has_executed = True
        if self.future is not None:
            self.consumed += 1
        return action

    def close(self) -> None:
        self.queue.clear()
        if self.future is not None:
            self.future.cancel()
        self.executor.shutdown(wait=True, cancel_futures=True)
