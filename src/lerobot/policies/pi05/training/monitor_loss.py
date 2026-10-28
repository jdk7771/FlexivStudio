#!/usr/bin/env python3
"""Monitor training progress and loss without modifying the running trainer.

The current local launcher writes metrics to the terminal only when wandb is
disabled. This utility samples rank 0 with py-spy and persists training and
validation loss in separate CSV files. Missing history is shown as a gap.
"""

from __future__ import annotations

import argparse
import copy
import csv
import http.server
import json
import math
import os
import re
import shutil
import socketserver
import subprocess
import sys
import threading
import time
import urllib.parse
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path


DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "checkpoints/DualArm_FoldTowel_pi05_rtc_full_finetune"
STEP_RE = re.compile(r"^\s+step: (\d+)\s*$", re.MULTILINE)
EVAL_LOSS_RE = re.compile(r"^\s+eval_loss: ([0-9.eE+-]+)\s*$", re.MULTILINE)
SCALAR_LOSS_RE = re.compile(r'"loss":\s*([0-9.eE+-]+)(?=[,}])')

DASHBOARD_HTML = Path(__file__).with_name("loss_dashboard.html").read_text(encoding="utf-8")


@dataclass(frozen=True)
class LossPoint:
    step: int
    loss: float
    captured_at: str
    source: str = ""


@dataclass(frozen=True)
class ProcessSample:
    step: int
    eval_loss: float | None
    train_loss: float | None
    train_loss_step: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pid", type=int, help="Rank-0 worker PID; discovered automatically by default")
    parser.add_argument("--interval", type=float, default=30.0, help="Polling interval in seconds")
    parser.add_argument("--once", action="store_true", help="Take one sample and exit")
    parser.add_argument("--serve", action="store_true", help="Serve the live dashboard while monitoring")
    parser.add_argument("--host", default="0.0.0.0", help="Dashboard listen address")
    parser.add_argument("--port", type=int, default=8766, help="Dashboard listen port")
    args = parser.parse_args()
    if not math.isfinite(args.interval) or args.interval <= 0:
        parser.error("--interval must be a finite positive number")
    if args.once and args.serve:
        parser.error("--once and --serve cannot be used together")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    return args


def read_cmdline(pid: int) -> list[str]:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return []
    return [part.decode(errors="replace") for part in raw.split(b"\0") if part]


def read_environ(pid: int) -> dict[str, str]:
    try:
        raw = Path(f"/proc/{pid}/environ").read_bytes()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return {}
    result = {}
    for part in raw.split(b"\0"):
        key, separator, value = part.partition(b"=")
        if separator:
            result[key.decode(errors="replace")] = value.decode(errors="replace")
    return result


def config_matches_output(args: list[str], output_dir: Path) -> bool:
    expected = output_dir.resolve()
    for argument in args:
        if argument.startswith("--output_dir="):
            return Path(argument.split("=", 1)[1]).resolve() == expected
        if not argument.startswith("--config_path="):
            continue
        config_path = Path(argument.split("=", 1)[1])
        try:
            config = json.loads(config_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            continue
        configured = config.get("output_dir")
        if configured and Path(configured).resolve() == expected:
            return True
    return False


def training_job_from_process(pid: int) -> tuple[Path, dict] | None:
    command = read_cmdline(pid)
    if "lerobot.scripts.lerobot_train" not in command:
        return None
    environment = read_environ(pid)
    rank = environment.get("RANK", environment.get("LOCAL_RANK"))
    if rank not in {None, "0"}:
        return None
    if rank is None and "torch.distributed.run" in command:
        return None

    config = {}
    output_dir = None
    for argument in command:
        if argument.startswith("--config_path="):
            try:
                config = json.loads(Path(argument.split("=", 1)[1]).read_text())
            except (FileNotFoundError, json.JSONDecodeError, OSError):
                config = {}
        elif argument.startswith("--output_dir="):
            output_dir = Path(argument.split("=", 1)[1])
    config = apply_cli_overrides(config, command)
    if output_dir is None and config.get("output_dir"):
        output_dir = Path(config["output_dir"])
    if output_dir is None:
        return None
    return output_dir.resolve(), config


def apply_cli_overrides(config: dict, command: list[str]) -> dict:
    """Use the running command's overrides, including those absent from checkpoints."""
    config = copy.deepcopy(config)
    config_path = None
    for index, argument in enumerate(command):
        if not argument.startswith("--"):
            continue
        name, separator, raw = argument[2:].partition("=")
        if not separator:
            if index + 1 >= len(command) or command[index + 1].startswith("--"):
                continue
            raw = command[index + 1]
        if name == "config_path":
            config_path = Path(raw)
            continue
        try:
            value = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            value = raw
        target = config
        keys = name.split(".")
        for key in keys[:-1]:
            if not isinstance(target.get(key), dict):
                target[key] = {}
            target = target[key]
        target[keys[-1]] = value
    if config.get("resume") and config_path and config_path.parent.name == "pretrained_model":
        config["checkpoint_path"] = str(config_path.parent.parent)
    return config


def effective_config(output_dir: Path, active: tuple[int, dict] | None) -> dict:
    return active[1] if active and active[1] else load_training_config(output_dir)


@lru_cache(maxsize=128)
def dataset_counts(root: str, episodes: tuple | None, excluded: tuple, split: float, metadata_mtime: int) -> dict:
    """Match episode selection and per-task splitting without loading images or videos."""
    try:
        info = json.loads((Path(root) / "meta/info.json").read_text())
        selected = set(episodes if episodes is not None else range(int(info["total_episodes"])))
        selected.difference_update(excluded)
        if split == 0:
            return {"train": len(selected), "eval": 0, "total": len(selected)}
        if int(info.get("total_tasks", 1)) == 1:
            n_eval = math.ceil(len(selected) * split)
        else:
            import pyarrow.parquet as pq

            task_counts: dict[str, int] = {}
            for path in sorted((Path(root) / "meta/episodes").rglob("*.parquet")):
                for row in pq.read_table(path, columns=["episode_index", "tasks"]).to_pylist():
                    if row["episode_index"] in selected:
                        task = row["tasks"][0] if row["tasks"] else ""
                        task_counts[task] = task_counts.get(task, 0) + 1
            if sum(task_counts.values()) != len(selected):
                return {"train": None, "eval": None, "total": len(selected)}
            n_eval = sum(math.ceil(count * split) for count in task_counts.values())
        return {"train": len(selected) - n_eval, "eval": n_eval, "total": len(selected)}
    except (OSError, ValueError, KeyError, ImportError):
        return {"train": None, "eval": None, "total": None}


def get_dataset_counts(config: dict) -> dict:
    dataset = config.get("dataset", {})
    root = dataset.get("root")
    if not root:
        return {"train": None, "eval": None, "total": None}
    try:
        mtime = (Path(root) / "meta/info.json").stat().st_mtime_ns
    except OSError:
        mtime = 0
    episodes = dataset.get("episodes")
    return dataset_counts(str(root), tuple(episodes) if episodes is not None else None,
                          tuple(dataset.get("exclude_episodes") or ()), float(dataset.get("eval_split", 0)), mtime)


def discover_active_jobs() -> dict[Path, tuple[int, dict]]:
    jobs = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        job = training_job_from_process(pid)
        if job is not None:
            output_dir, config = job
            jobs[output_dir] = (pid, config)
    return jobs


def discover_rank_zero(output_dir: Path) -> int | None:
    job = discover_active_jobs().get(output_dir.resolve())
    return job[0] if job else None


def load_training_config(output_dir: Path) -> dict:
    checkpoints = output_dir / "checkpoints"
    last = checkpoints / "last" / "pretrained_model" / "train_config.json"
    candidates = [last]
    if checkpoints.is_dir():
        candidates.extend(
            path / "pretrained_model" / "train_config.json"
            for path in sorted(checkpoints.iterdir(), reverse=True)
            if path.is_dir() and path.name.isdigit()
        )
    for path in candidates:
        try:
            return json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            continue
    return {}


def latest_checkpoint_step(output_dir: Path) -> int:
    checkpoints = output_dir / "checkpoints"
    if not checkpoints.is_dir():
        return 0
    return max((int(path.name) for path in checkpoints.iterdir() if path.is_dir() and path.name.isdigit()), default=0)


def known_output_dirs(outputs_root: Path, active_jobs: dict[Path, tuple[int, dict]]) -> list[Path]:
    directories = set(active_jobs)
    if outputs_root.is_dir():
        for path in outputs_root.iterdir():
            if path.is_dir() and ((path / "checkpoints").is_dir() or (path / "training_status.json").exists()):
                directories.add(path.resolve())
    return list(directories)


def read_status(output_dir: Path, active_jobs: dict[Path, tuple[int, dict]], interval: float) -> dict:
    status_path = output_dir / "training_status.json"
    try:
        status = json.loads(status_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        status = {}

    active = active_jobs.get(output_dir)
    config = effective_config(output_dir, active)
    step = max(int(status.get("step") or 0), latest_checkpoint_step(output_dir))
    total_steps = int(config.get("steps", status.get("total_steps", 0)) or 0)
    points = read_points(output_dir / "eval_loss_monitor.csv")
    latest = points[-1] if points else None
    train_points = read_points(output_dir / "train_loss_monitor.csv", "train_loss")
    latest_train = train_points[-1] if train_points else None
    status.update(
        {
            "running": active is not None,
            "pid": active[0] if active else None,
            "job_name": config.get("job_name") or status.get("job_name") or output_dir.name,
            "step": step,
            "total_steps": total_steps,
            "progress": 100 * step / total_steps if total_steps else None,
            "eval_loss": latest.loss if latest else status.get("eval_loss"),
            "eval_loss_step": latest.step if latest else status.get("eval_loss_step"),
            "eval_interval": int(config.get("eval_steps", status.get("eval_interval", 0)) or 0),
            "train_loss": latest_train.loss if latest_train else None,
            "train_loss_step": latest_train.step if latest_train else None,
            "train_loss_source": latest_train.source if latest_train else None,
            "dataset_counts": get_dataset_counts(config),
            "train_loss_points": len(train_points),
            "history_gaps": history_gaps(train_points, step, int(config.get("log_freq", 50) or 50)),
            "gpus": read_environ(active[0]).get("CUDA_VISIBLE_DEVICES", "unknown") if active else status.get("gpus"),
            "interval": interval,
            "config": config or status.get("config", {}),
        }
    )
    return status


def list_jobs(outputs_root: Path, interval: float) -> list[dict]:
    active_jobs = discover_active_jobs()
    jobs = []
    for output_dir in known_output_dirs(outputs_root, active_jobs):
        status = read_status(output_dir, active_jobs, interval)
        try:
            modified = output_dir.stat().st_mtime
        except OSError:
            modified = 0.0
        jobs.append(
            {
                "job_name": status["job_name"],
                "output_dir": str(output_dir),
                "running": status["running"],
                "step": status["step"],
                "total_steps": status["total_steps"],
                "modified": modified,
            }
        )
    jobs.sort(key=lambda job: (not job["running"], -job["modified"], job["job_name"]))
    return jobs


def parse_process_sample(output: str) -> ProcessSample:
    # Scope variables to the training-loop frame: other frames can also have a step or loss.
    frames = re.split(r"(?=^    \S.* \([^\n]+:\d+\)\s*$)", output, flags=re.MULTILINE)
    train_frame = next((frame for frame in frames if re.match(r"    train \(.*lerobot_train.py:", frame)), "")
    step_matches = STEP_RE.findall(train_frame)
    if not step_matches:
        raise RuntimeError("py-spy did not expose the training-loop step")
    step = int(step_matches[-1])
    eval_matches = EVAL_LOSS_RE.findall(train_frame)
    update_frame = next((frame for frame in frames if re.match(r"    update_policy \(.*lerobot_train.py:", frame)), "")
    # output_dict belongs to this batch only after policy() has returned. Ignore a retained
    # dictionary while the next forward is in progress, and pair it with the pending update.
    inner_frames = output[:output.find(update_frame)] if update_frame else ""
    in_backward = bool(re.search(r"^    (?:backward|_engine_run_backward) \(", inner_frames, re.MULTILINE))
    forward_pending = not in_backward and bool(re.search(r"^    forward \(", inner_frames, re.MULTILINE))
    train_loss = None
    if update_frame and not forward_pending:
        output_dict = re.search(r"^\s+output_dict: (.*)$", update_frame, re.MULTILINE)
        if output_dict:
            match = SCALAR_LOSS_RE.search(output_dict[1])
            if match and math.isfinite(float(match[1])):
                train_loss = float(match[1])
    return ProcessSample(step, float(eval_matches[-1]) if eval_matches else None,
                         train_loss, step + 1 if train_loss is not None else None)


def sample_process(pid: int, py_spy: str) -> ProcessSample:
    result = subprocess.run(
        [py_spy, "dump", "--nonblocking", "-llll", "--pid", str(pid)],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=15,
    )
    output = result.stdout + result.stderr
    if result.returncode != 0:
        detail = output.strip().splitlines()[-1] if output.strip() else "unknown py-spy error"
        raise RuntimeError(detail)
    return parse_process_sample(output)


def read_points(path: Path, loss_key: str = "eval_loss") -> list[LossPoint]:
    if not path.exists():
        return []
    points = []
    with path.open(newline="", encoding="utf-8", errors="replace") as stream:
        for row in csv.DictReader(stream):
            try:
                loss = float(row[loss_key])
                if math.isfinite(loss):
                    points.append(LossPoint(int(row["step"]), loss, row["captured_at"], row.get("source", "")))
            except (KeyError, TypeError, ValueError):
                continue
    return sorted({point.step: point for point in points}.values(), key=lambda point: point.step)


def write_points(path: Path, points: list[LossPoint], loss_key: str = "eval_loss") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("step", loss_key, "captured_at", "source"))
        writer.writeheader()
        for point in points:
            writer.writerow({"step": point.step, loss_key: f"{point.loss:.9g}", "captured_at": point.captured_at, "source": point.source})
    temporary.replace(path)


def history_gaps(points: list[LossPoint], step: int, log_freq: int) -> list[dict]:
    gaps = []
    if not points:
        return [{"from": 0, "to": step}] if step > 0 else []
    if points[0].step > max(1, log_freq):
        gaps.append({"from": 0, "to": points[0].step})
    threshold = max(150, log_freq * 3)
    for previous, current in zip(points, points[1:]):
        if current.step - previous.step > threshold:
            gaps.append({"from": previous.step, "to": current.step})
    if step - points[-1].step > threshold:
        gaps.append({"from": points[-1].step, "to": step})
    return gaps


def write_svg(path: Path, points: list[LossPoint]) -> None:
    if not points:
        return
    width, height = 960, 540
    left, right, top, bottom = 90, 35, 45, 70
    plot_width, plot_height = width - left - right, height - top - bottom
    min_step, max_step = points[0].step, points[-1].step
    min_loss = min(point.loss for point in points)
    max_loss = max(point.loss for point in points)
    if min_step == max_step:
        min_step, max_step = 0, max(1, max_step)
    if min_loss == max_loss:
        padding = max(abs(min_loss) * 0.05, 0.001)
        min_loss -= padding
        max_loss += padding

    def x(step: int) -> float:
        return left + (step - min_step) / (max_step - min_step) * plot_width

    def y(loss: float) -> float:
        return top + (max_loss - loss) / (max_loss - min_loss) * plot_height

    polyline = " ".join(f"{x(point.step):.1f},{y(point.loss):.1f}" for point in points)
    circles = "".join(
        f'<circle cx="{x(point.step):.1f}" cy="{y(point.loss):.1f}" r="4" fill="#c23b22"/>'
        for point in points
    )
    grid = []
    labels = []
    for index in range(6):
        fraction = index / 5
        grid_y = top + fraction * plot_height
        value = max_loss - fraction * (max_loss - min_loss)
        grid.append(f'<line x1="{left}" y1="{grid_y:.1f}" x2="{width-right}" y2="{grid_y:.1f}"/>')
        labels.append(f'<text x="{left-12}" y="{grid_y+5:.1f}" text-anchor="end">{value:.4f}</text>')
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#ffffff"/>
<text x="{left}" y="28" font-family="sans-serif" font-size="20" font-weight="600">PI0.5 validation loss</text>
<g stroke="#d9d9d9" stroke-width="1">{''.join(grid)}</g>
<g font-family="sans-serif" font-size="13" fill="#444">{''.join(labels)}</g>
<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#333"/>
<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#333"/>
<polyline points="{polyline}" fill="none" stroke="#c23b22" stroke-width="3" stroke-linejoin="round"/>
{circles}
<text x="{left + plot_width/2:.1f}" y="{height-20}" text-anchor="middle" font-family="sans-serif" font-size="14">training step ({min_step} to {max_step})</text>
<text x="22" y="{top + plot_height/2:.1f}" text-anchor="middle" transform="rotate(-90 22 {top + plot_height/2:.1f})" font-family="sans-serif" font-size="14">eval loss</text>
</svg>
'''
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(svg)
    temporary.replace(path)


def resumed_from_step(config: dict) -> int:
    checkpoint_path = config.get("checkpoint_path")
    if not checkpoint_path:
        return 0
    checkpoint_name = Path(checkpoint_path).name
    return int(checkpoint_name) if checkpoint_name.isdigit() else 0


def format_eta(step: int, total_steps: int, started_at: float | None, initial_step: int = 0) -> str:
    completed = step - initial_step
    if not started_at or completed <= 0 or step >= total_steps:
        return "unknown" if step < total_steps else "complete"
    elapsed = time.time() - started_at
    remaining = elapsed / completed * (total_steps - step)
    eta = datetime.fromtimestamp(time.time() + remaining).astimezone()
    return f"{remaining / 3600:.1f}h ({eta:%m-%d %H:%M})"


def write_status(
    path: Path,
    *,
    config: dict,
    pid: int,
    step: int,
    eval_loss: float | None,
    eval_step: int | None,
    interval: float,
    started_at: float | None,
) -> None:
    total_steps = int(config["steps"])
    environment = read_environ(pid)
    payload = {
        "running": Path(f"/proc/{pid}").exists(),
        "pid": pid,
        "job_name": config.get("job_name"),
        "step": step,
        "total_steps": total_steps,
        "progress": 100 * step / total_steps,
        "eval_loss": eval_loss,
        "eval_loss_step": eval_step,
        "eval_interval": int(config["eval_steps"]),
        "eta": format_eta(step, total_steps, started_at, resumed_from_step(config)),
        "gpus": environment.get("CUDA_VISIBLE_DEVICES", "unknown"),
        "interval": interval,
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "config": config,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    temporary.replace(path)


class ReusableThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def start_dashboard(
    outputs_root: Path, default_output: Path, host: str, port: int, interval: float
) -> ReusableThreadingHTTPServer:
    class DashboardHandler(http.server.BaseHTTPRequestHandler):
        def send_content(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urllib.parse.urlsplit(self.path)
            route = parsed.path
            if route in {"/", "/index.html"}:
                self.send_content(200, "text/html; charset=utf-8", DASHBOARD_HTML.encode())
                return
            jobs = list_jobs(outputs_root, interval)
            if route == "/api/jobs":
                self.send_content(200, "application/json", json.dumps(jobs, ensure_ascii=False).encode())
                return
            query = urllib.parse.parse_qs(parsed.query)
            requested = Path(query.get("output", [str(default_output)])[0]).resolve()
            allowed = {Path(job["output_dir"]): job for job in jobs}
            if requested not in allowed:
                self.send_content(404, "text/plain; charset=utf-8", b"Unknown training output")
                return
            if route == "/api/status":
                status = read_status(requested, discover_active_jobs(), interval)
                self.send_content(200, "application/json", json.dumps(status, ensure_ascii=False).encode())
                return
            if route == "/api/loss":
                kind = query.get("kind", ["train"])[0]
                if kind not in {"train", "eval"}:
                    self.send_content(400, "text/plain; charset=utf-8", b"Unknown loss kind")
                    return
                points = read_points(requested / f"{kind}_loss_monitor.csv", f"{kind}_loss")
                body = json.dumps([{"step": point.step, "loss": point.loss, "source": point.source,
                                    "captured_at": point.captured_at} for point in points]).encode()
                self.send_content(200, "application/json", body)
                return
            self.send_content(404, "text/plain; charset=utf-8", b"Not found")

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ReusableThreadingHTTPServer((host, port), DashboardHandler)
    threading.Thread(target=server.serve_forever, name="loss-dashboard", daemon=True).start()
    return server


def process_started_at(pid: int) -> float | None:
    try:
        uptime = float(Path("/proc/uptime").read_text().split()[0])
        fields = Path(f"/proc/{pid}/stat").read_text().split()
        ticks = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
        return time.time() - uptime + int(fields[21]) / ticks
    except (FileNotFoundError, OSError, ValueError, IndexError):
        return None


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    outputs_root = output_dir.parent
    py_spy = shutil.which("py-spy")
    if py_spy is None:
        print("py-spy is required to monitor a run that was started without a log file.", file=sys.stderr)
        return 2

    server = None
    if args.serve:
        server = start_dashboard(outputs_root, output_dir, args.host, args.port, args.interval)
        browser_host = "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host
        print(f"Dashboard: http://{browser_host}:{args.port}/")
        print(f"Watching active training jobs under {outputs_root}")

    started_at: dict[int, float | None] = {}
    waiting_reported = False

    try:
        while True:
            active_jobs = discover_active_jobs()
            if args.pid:
                config = load_training_config(output_dir)
                active_jobs = {output_dir: (args.pid, config)}
            jobs_to_sample = active_jobs if args.serve else {
                output_dir: active_jobs[output_dir]
            } if output_dir in active_jobs else {}

            if not jobs_to_sample:
                if args.once:
                    print(f"No rank-0 training process found for {output_dir}", file=sys.stderr)
                    return 1
                if not args.serve:
                    print("Training process exited.")
                    return 0
                if not waiting_reported:
                    print("No active training jobs; dashboard remains available.")
                    waiting_reported = True

            for job_output, (pid, source_config) in jobs_to_sample.items():
                config = source_config or load_training_config(job_output)
                total_steps = int(config.get("steps", 0))
                eval_interval = int(config.get("eval_steps", 0))
                if total_steps <= 0:
                    print(f"Skipping {job_output}: steps unavailable", file=sys.stderr)
                    continue
                try:
                    sample = sample_process(pid, py_spy)
                    for attempt in range(2):
                        if sample.train_loss is not None:
                            break
                        time.sleep(0.25)
                        sample = sample_process(pid, py_spy)
                    step, eval_loss = sample.step, sample.eval_loss
                except (RuntimeError, OSError, ValueError, subprocess.TimeoutExpired) as error:
                    print(f"Sample unavailable for PID {pid}: {error}", file=sys.stderr)
                    if args.once:
                        return 1
                    continue

                csv_path = job_output / "eval_loss_monitor.csv"
                svg_path = job_output / "eval_loss_curve.svg"
                status_path = job_output / "training_status.json"
                if sample.train_loss is not None and sample.train_loss_step is not None:
                    train_csv = job_output / "train_loss_monitor.csv"
                    train_points = {point.step: point for point in read_points(train_csv, "train_loss")}
                    existing = train_points.get(sample.train_loss_step)
                    if existing is None or existing.source != "terminal_window_mean":
                        train_points[sample.train_loss_step] = LossPoint(
                            sample.train_loss_step, sample.train_loss,
                            datetime.now().astimezone().isoformat(timespec="seconds"), "rank0_batch_sample",
                        )
                    write_points(train_csv, sorted(train_points.values(), key=lambda point: point.step), "train_loss")
                eval_step = step // eval_interval * eval_interval if eval_interval > 0 else 0
                points = read_points(csv_path)
                if eval_loss is not None and eval_step > 0:
                    by_step = {point.step: point for point in points}
                    by_step[eval_step] = LossPoint(
                        eval_step,
                        eval_loss,
                        datetime.now().astimezone().isoformat(timespec="seconds"),
                    )
                    points = sorted(by_step.values(), key=lambda point: point.step)
                    write_points(csv_path, points)
                    write_svg(svg_path, points)
                latest = points[-1] if points else None
                if pid not in started_at:
                    started_at[pid] = process_started_at(pid)
                write_status(
                    status_path,
                    config=config,
                    pid=pid,
                    step=step,
                    eval_loss=latest.loss if latest else None,
                    eval_step=latest.step if latest else None,
                    interval=args.interval,
                    started_at=started_at[pid],
                )
                latest_text = (
                    f"train_loss={sample.train_loss:.6f} @ {sample.train_loss_step}"
                    if sample.train_loss is not None else "train_loss=waiting for batch sample"
                )
                print(
                    f"{datetime.now().astimezone():%H:%M:%S}  {config.get('job_name', job_output.name)}  "
                    f"step={step}/{total_steps} ({100 * step / total_steps:.1f}%)  {latest_text}  "
                    f"ETA={format_eta(step, total_steps, started_at[pid], resumed_from_step(config))}",
                    flush=True,
                )
                waiting_reported = False
            if args.once:
                return 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nMonitor stopped; training was not interrupted.")
        return 0
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    raise SystemExit(main())
