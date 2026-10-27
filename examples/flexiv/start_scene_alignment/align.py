"""Standalone top-camera reference capture and manual overlay alignment.

No model, robot SDK, deployment module or deployment configuration is imported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any
from uuid import uuid4

import numpy as np
from PIL import Image

LOGGER = logging.getLogger(__name__)
SCENES = tuple(f"test_{index:02d}" for index in range(1, 11))
MAX_AGE_S = 0.3
DEFAULT_DIRECTORY = Path.home() / "snap/deployment_logs/start_scenes"


def validate_rgb(rgb: np.ndarray, camera: dict) -> None:
    if rgb.shape != (camera["height"], camera["width"], 3) or rgb.dtype != np.uint8:
        raise ValueError("Expected an original-resolution RGB uint8 frame")


class ReferenceStore:
    def __init__(self, directory: Path, camera: dict) -> None:
        self.directory = directory
        self.camera = camera

    def saved(self) -> list[str]:
        return [name for name in SCENES if (self.directory / name / "reference.json").is_file()]

    def load(self, name: str) -> tuple[dict, np.ndarray]:
        if name not in SCENES:
            raise ValueError("Unknown scene")
        path = self.directory / name
        meta = json.loads((path / "reference.json").read_text(encoding="utf-8"))
        if meta.get("schema_version") != 1 or meta.get("scene") != name:
            raise ValueError(f"{name}: invalid reference metadata")
        if meta.get("camera") != self.camera:
            raise ValueError(f"{name}: camera serial, resolution or FPS differs from the reference")
        if hashlib.sha256((path / "top.png").read_bytes()).hexdigest() != meta["image_sha256"]:
            raise ValueError(f"{name}: reference image checksum differs")
        with Image.open(path / "top.png") as image:
            rgb = np.asarray(image.convert("RGB")).copy()
        validate_rgb(rgb, self.camera)
        return meta, rgb

    def save(self, name: str, rgb: np.ndarray, received: float, note: str) -> Path:
        if name not in SCENES:
            raise ValueError("Unknown scene")
        validate_rgb(rgb, self.camera)
        # All ten references belong to the same camera setup.
        for existing in self.saved():
            self.load(existing)
        self.directory.mkdir(parents=True, exist_ok=True)
        destination = self.directory / name
        if destination.exists():
            raise FileExistsError(f"{name} already exists; choose another slot or a new --directory")
        staging = self.directory / f".{name}-{uuid4().hex}"
        staging.mkdir()
        try:
            Image.fromarray(rgb).save(staging / "top.png")
            meta = {
                "schema_version": 1,
                "scene": name,
                "camera": self.camera,
                "wall_time": time.time(),
                "received_monotonic_s": received,
                "note": note,
                "image_sha256": hashlib.sha256((staging / "top.png").read_bytes()).hexdigest(),
                "scope": "Top RGB only; robot state and physical camera pose are not measured",
            }
            (staging / "reference.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
            staging.rename(destination)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        return destination

    def save_confirmation(
        self, name: str, reference: tuple[dict, np.ndarray], rgb: np.ndarray, received: float, note: str
    ) -> Path:
        validate_rgb(rgb, self.camera)
        meta, target = reference
        if name != meta["scene"]:
            raise ValueError("Reference scene changed")
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        parent = self.directory / "checks"
        parent.mkdir(parents=True, exist_ok=True)
        destination = parent / f"{stamp}_{name}_{uuid4().hex[:8]}"
        staging = parent / f".{destination.name}"
        staging.mkdir()
        try:
            Image.fromarray(target).save(staging / "reference_top.png")
            Image.fromarray(rgb).save(staging / "actual_top.png")
            Image.blend(Image.fromarray(target), Image.fromarray(rgb), 0.5).save(staging / "overlay.png")
            audit = {
                "scene": name,
                "reference": meta,
                "camera": self.camera,
                "wall_time": time.time(),
                "received_monotonic_s": received,
                "note": note,
                "confirmation": "Manual overlay confirmation, not an automatic similarity pass",
                "robot_state_verified": False,
                "deployment_started": False,
            }
            (staging / "alignment.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
            staging.rename(destination)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        return destination


class TopCamera:
    """One RealSense color stream; newest frame replaces the previous one."""

    def __init__(self, camera: dict) -> None:
        import pyrealsense2 as rs

        self._pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(camera["serial"])
        config.enable_stream(
            rs.stream.color, camera["width"], camera["height"], rs.format.rgb8, camera["fps"]
        )
        self._lock, self._stop = Lock(), Event()
        self._latest: tuple[np.ndarray, float] | None = None
        self._error: Exception | None = None
        self._closed = False
        self._pipeline.start(config)
        self._thread = Thread(target=self._read, daemon=True, name="start-scene-top-camera")
        try:
            self._thread.start()
        except BaseException:
            self._pipeline.stop()
            raise

    def _read(self) -> None:
        try:
            while not self._stop.is_set():
                frames = self._pipeline.poll_for_frames()
                if not frames:
                    self._stop.wait(0.005)
                    continue
                color = frames.get_color_frame()
                if color:
                    received = time.monotonic()
                    rgb = np.asanyarray(color.get_data()).copy()
                    with self._lock:
                        self._latest = rgb, received
        except Exception as exc:
            with self._lock:
                self._error = exc

    def latest(self) -> tuple[np.ndarray, float] | None:
        with self._lock:
            if self._error:
                raise RuntimeError(f"Top camera reader failed: {self._error}") from self._error
            return self._latest

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        try:
            self._pipeline.stop()
        finally:
            self._thread.join(timeout=2)


def fresh_frame(frame: tuple[np.ndarray, float] | None) -> bool:
    return frame is not None and 0 <= time.monotonic() - frame[1] <= MAX_AGE_S


class AlignmentWindow:
    def __init__(self, store: ReferenceStore, *, capture: bool, scene: str | None, note: str) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.store, self.capture = store, capture
        names = list(SCENES) if capture else store.saved()
        if not names:
            raise ValueError("No saved references; run with --capture first")
        if scene is not None and scene not in names:
            raise ValueError(f"{scene} has not been captured")
        self.camera: Any = None
        self.error: BaseException | None = None
        self.reference: tuple[dict, np.ndarray] | None = None
        self.displayed: tuple[np.ndarray, float] | None = None
        self.result: Path | None = None
        self._destroyed = False
        self.root = tk.Tk()
        self.root.title("起始场景：采集 10 个参考" if capture else "起始场景：独立叠图对齐")
        self.root.report_callback_exception = self.callback_error
        self.root.protocol("WM_DELETE_WINDOW", self.root.quit)
        initial = scene or next((name for name in names if name not in store.saved()), names[0])
        self.scene = tk.StringVar(value=initial)
        selector = ttk.Combobox(self.root, textvariable=self.scene, values=names, state="readonly", width=22)
        selector.pack()
        selector.bind("<<ComboboxSelected>>", lambda _: self.select())
        self.message = tk.StringVar()
        ttk.Label(self.root, textvariable=self.message, wraplength=1000).pack()
        self.preview = ttk.Label(self.root)
        self.preview.pack()
        self.alpha = tk.DoubleVar(value=0.5)
        ttk.Label(self.root, text="透明度：0 = 参考图，1 = 实时图；请同时对齐桌面固定标记和毛巾轮廓").pack()
        ttk.Scale(self.root, from_=0, to=1, variable=self.alpha, length=650).pack()
        self.confirm_visual = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            self.root,
            variable=self.confirm_visual,
            text="我已统一机械臂/夹爪姿态，确认场景摆放，并将手移出画面（程序不检测机械臂状态）",
        ).pack()
        self.note = tk.StringVar(value=note)
        ttk.Label(self.root, text="备注（如模型、RTC 开关、重复次数）：").pack()
        ttk.Entry(self.root, textvariable=self.note, width=75).pack()
        self.button = ttk.Button(
            self.root,
            command=self.confirm,
            text="保存参考并进入下一个 / Enter" if capture else "保存对齐结果并退出 / Enter",
        )
        self.button.pack()
        ttk.Button(self.root, text="结束采集" if capture else "取消", command=self.root.quit).pack()
        self.root.bind("<Return>", lambda _: self.confirm())
        self.root.bind("<Escape>", lambda _: self.root.quit())
        try:
            self.select()
        except BaseException:
            self.close()
            raise

    def callback_error(self, exc_type: Any, exc: BaseException, traceback: Any) -> None:
        self.error = exc
        self.root.quit()

    def select(self) -> None:
        self.displayed = None
        self.confirm_visual.set(False)
        self.reference = None
        if self.scene.get() in self.store.saved():
            self.reference = self.store.load(self.scene.get())
        self.button.configure(state="disabled")

    def update(self) -> None:
        from PIL import ImageTk

        try:
            frame = self.camera.latest()
            self.displayed = frame
            ready = fresh_frame(frame)
            occupied = self.capture and (self.store.directory / self.scene.get()).exists()
            if frame is not None:
                rgb, _ = frame
                validate_rgb(rgb, self.store.camera)
                live = Image.fromarray(rgb)
                display = (
                    live
                    if self.reference is None
                    else Image.blend(Image.fromarray(self.reference[1]), live, float(self.alpha.get()))
                )
                display.thumbnail(
                    (
                        min(1100, self.root.winfo_screenwidth() - 80),
                        max(180, self.root.winfo_screenheight() - 320),
                    )
                )
                self.photo = ImageTk.PhotoImage(display)
                self.preview.configure(image=self.photo)
            self.button.configure(state="normal" if ready and not occupied else "disabled")
            status = (
                "此编号已保存，不会覆盖"
                if occupied
                else ("请人工确认后保存；不会自动启动部署" if ready else "等待新鲜相机画面，禁止保存")
            )
            self.message.set(f"{self.scene.get()} | 已采集 {len(self.store.saved())}/10 | {status}")
        except BaseException as exc:
            self.error = exc
            self.root.quit()
            return
        self.root.after(66, self.update)

    def confirm(self) -> None:
        if not self.confirm_visual.get():
            self.message.set("请先勾选场景与机械臂姿态的人工确认。")
            return
        try:
            # Save exactly the frame displayed to the operator, never an unseen newer frame.
            if not fresh_frame(self.displayed) or not fresh_frame(self.camera.latest()):
                self.message.set("画面过期，请等待新画面后再确认。")
                return
            assert self.displayed is not None
            rgb, received = self.displayed
            name = self.scene.get()
            if self.capture:
                if (self.store.directory / name).exists():
                    self.message.set("该编号已保存，请选择空编号或新的参考目录。")
                    return
                self.result = self.store.save(name, rgb, received, self.note.get())
                LOGGER.info("Saved reference: %s", self.result)
                remaining = [item for item in SCENES if not (self.store.directory / item).exists()]
                if not remaining:
                    self.root.quit()
                    return
                self.scene.set(remaining[0])
                self.select()
            else:
                assert self.reference is not None
                self.result = self.store.save_confirmation(
                    name, self.reference, rgb, received, self.note.get()
                )
                LOGGER.info(
                    "Saved manual alignment: %s; camera will close; start deployment separately", self.result
                )
                self.root.quit()
        except BaseException as exc:
            self.error = exc
            self.root.quit()

    def run(self, camera: Any) -> None:
        self.camera = camera
        self.root.after(0, self.update)
        try:
            self.root.mainloop()
            if self.error is not None:
                raise self.error
        finally:
            self.close()

    def close(self) -> None:
        if not self._destroyed:
            self._destroyed = True
            self.root.destroy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--capture", action="store_true", help="Capture up to ten reference scenes")
    mode.add_argument(
        "--compare", nargs="?", const="choose", choices=("choose", *SCENES), help="Overlay a saved reference"
    )
    mode.add_argument("--list", action="store_true", help="Validate/list references; no camera or window")
    parser.add_argument("--directory", type=Path, default=DEFAULT_DIRECTORY)
    parser.add_argument("--serial", default="350122061994", help="Top RealSense serial")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, choices=(6, 15, 30), default=15)
    parser.add_argument("--note", default="")
    args = parser.parse_args()
    if args.width <= 0 or args.height <= 0:
        parser.error("Image width and height must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    store = ReferenceStore(
        args.directory.expanduser().resolve(),
        {
            "serial": args.serial,
            "width": args.width,
            "height": args.height,
            "fps": args.fps,
        },
    )
    for name in store.saved():
        store.load(name)
    if args.list:
        for name in SCENES:
            print(f"{name}: {'saved / valid' if name in store.saved() else 'not captured'}")
        return
    window = AlignmentWindow(
        store, capture=args.capture, scene=None if args.compare == "choose" else args.compare, note=args.note
    )
    camera = None
    try:
        camera = TopCamera(store.camera)
        window.run(camera)
    finally:
        try:
            if camera is not None:
                camera.close()
        finally:
            window.close()


if __name__ == "__main__":
    main()
