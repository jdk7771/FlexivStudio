"""Offline regression tests: transforms, RTC scheduling, prefix conditioning and shutdown."""

from __future__ import annotations

import copy
import importlib.util
import sys
from collections import deque
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
import shared

# isort: split

from flexiv_deploy_contract import ACTION_NAMES, CAMERA_NAMES, read_json

from contract import checkpoints, inspect
from engine import ChunkEngine
from policy import Transforms, sample_rtc


class Tokenizer:
    def __init__(self) -> None:
        self.prompts = None

    def __call__(self, prompts, **kwargs):
        self.prompts = prompts
        return SimpleNamespace(
            input_ids=torch.ones(len(prompts), 200, dtype=torch.int64),
            attention_mask=torch.ones(len(prompts), 200, dtype=torch.int64),
        )


@pytest.fixture
def config():
    return read_json(Path(__file__).with_name("deploy_config.json"))


@pytest.fixture
def training():
    train = read_json(shared.OPENPI_ROOT / "training/configs/dualarm_fold_towel_rtc_full.json")
    stats = {key: {"q01": [-1.0] * 16, "q99": [1.0] * 16} for key in ["action", "observation.state"]}
    stats["action"]["q99"][2] = stats["action"]["q01"][2]
    return train, stats


def test_transforms_match_training_collator(training):
    spec = importlib.util.spec_from_file_location(
        "openpi_training_runtime", shared.OPENPI_ROOT / "training/runtime.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    train, stats = training
    meta = SimpleNamespace(
        stats=stats,
        features={"action": {"names": ACTION_NAMES}},
        camera_keys=[f"observation.images.{n}" for n in CAMERA_NAMES],
    )
    collator = module.Collator(train, meta)
    collator.tokenizer = Tokenizer()
    state = np.linspace(-0.3, 0.5, 16, dtype=np.float32)
    actions = np.stack([state + 0.05, state - 0.03])
    sample = {
        "observation.state": torch.tensor(state),
        "action": torch.tensor(actions),
        "task": "DualArm_FoldTowel",
    }
    obs = dict(zip(ACTION_NAMES, state, strict=True))
    for i, n in enumerate(CAMERA_NAMES):
        image = np.full((224, 224, 3), i * 35, dtype=np.uint8)
        obs[n] = image
        sample[f"observation.images.{n}"] = torch.from_numpy(image).permute(2, 0, 1).float() / 255
    expected, expected_actions = collator([sample])
    tf = Transforms(train, stats, Tokenizer())
    actual, anchor = tf.observation(obs, "DualArm_FoldTowel", "cpu")
    assert tf.tokenizer.prompts == collator.tokenizer.prompts
    torch.testing.assert_close(actual.state, expected["state"])
    for key in actual.images:
        torch.testing.assert_close(actual.images[key], expected["images"][key])
    np.testing.assert_allclose(tf.encode_actions(actions, anchor), expected_actions[0].numpy(), rtol=1e-6)
    # Exclude the intentionally near-degenerate quantile channel from round-trip precision check.
    recovered = tf.decode_actions(tf.encode_actions(actions, anchor), anchor)
    np.testing.assert_allclose(recovered, actions, atol=2e-6)


def test_reanchor_keeps_grippers_absolute(training):
    tf = Transforms(*training, Tokenizer())
    absolute = np.full((8, 16), 0.1, dtype=np.float32)
    old_anchor = np.zeros(16, dtype=np.float32)
    new_anchor = np.full(16, 0.02, dtype=np.float32)
    a = tf.encode_actions(absolute, old_anchor)
    b = tf.encode_actions(absolute, new_anchor)
    np.testing.assert_array_equal(a[:, [7, 15]], b[:, [7, 15]])
    assert not np.array_equal(a[:, 0], b[:, 0])
    np.testing.assert_allclose(tf.decode_actions(b, new_anchor), absolute, atol=1e-6)


class Executor:
    def __init__(self) -> None:
        self.jobs: list = []

    def submit(self, fn, *args):
        future = Future()
        self.jobs.append((future, args))
        return future

    def shutdown(self, **kwargs):
        pass


def make_engine(config):
    executor = Executor()
    clock = [0.0]
    engine = ChunkEngine(
        SimpleNamespace(predict=lambda *a: None), config, executor=executor, clock=lambda: clock[0]
    )
    return engine, executor, clock


def chunk(offset=0):
    return np.repeat((np.arange(50, dtype=np.float32) + offset)[:, None], 16, axis=1)


def test_rtc_prefix_splice_without_replay(config):
    engine, ex, clock = make_engine(config)
    assert engine.tick({}, 0.0) is None
    ex.jobs[-1][0].set_result(chunk())
    clock[0] = 0.15
    assert engine.tick({}, 0.0)[0] == 0
    for i in range(1, 25):
        assert engine.tick({}, 0.0)[0] == i
    # 150 ms turnaround => ceil(150/66.67)=3, not the configured maximum of 8.
    assert engine.tick({}, 0.0)[0] == 25
    assert len(ex.jobs[-1][1][1]) == 3
    assert engine.tick({}, 0.0)[0] == 26
    assert engine.tick({}, 0.0)[0] == 27
    predicted = chunk(100)
    predicted[:3] = ex.jobs[-1][1][1]
    ex.jobs[-1][0].set_result(predicted)
    clock[0] = 0.30
    assert engine.tick({}, 0.0)[0] == 103
    assert engine.last_prefix_length == 3
    assert engine.last_trimmed == 3


def test_long_inference_discards_instead_of_splicing(config):
    engine, ex, clock = make_engine(config)
    engine.tick({}, 0.0)
    ex.jobs[-1][0].set_result(chunk())
    clock[0] = 0.115
    for i in range(28):
        assert engine.tick({}, 0.0)[0] == i
    assert engine.last_prefix_length == 2
    predicted = chunk(100)
    predicted[:2] = ex.jobs[-1][1][1]
    ex.jobs[-1][0].set_result(predicted)
    clock[0] = 0.32
    assert engine.tick({}, 0.0)[0] == 28  # Old queue, not the rejected 100+ targets.
    assert engine.discarded == 1
    assert engine.last_prefix_length == 4  # Adapt to 205 ms observed turnaround.


@pytest.mark.parametrize("elapsed,consumed", [(0.2, 2), (0.1, 3)])
def test_overlap_checks_time_and_actual_dispatch(config, elapsed, consumed):
    engine, ex, clock = make_engine(config)
    engine.queue.extend(chunk())
    engine.prefix = chunk()[:2]
    engine.consumed = consumed
    engine.future = Future()
    engine.future.set_result(chunk())
    clock[0] = elapsed
    engine._accept()
    assert engine.discarded == 1
    assert len(engine.queue) == 50


def test_five_discards_and_starvation_are_terminal(config):
    engine, ex, clock = make_engine(config)
    engine.queue.extend(chunk())
    engine.prefix = chunk()[:2]
    clock[0] = 0.3
    for _ in range(4):
        engine.future = Future()
        engine.future.set_result(chunk())
        engine._accept()
    engine.future = Future()
    engine.future.set_result(chunk())
    with pytest.raises(RuntimeError, match="5 results"):
        engine._accept()
    engine, ex, clock = make_engine(config)
    engine.has_executed = True
    with pytest.raises(RuntimeError, match="buffer exhausted"):
        engine.tick({}, 0.0)


def test_dynamic_prefix_refill_and_cold_start(config):
    engine, ex, clock = make_engine(config)
    assert engine.prefix_length() == 0
    engine.queue.extend(chunk())
    assert engine.prefix_length() == 8  # No latency measurement yet.
    engine.initial_turnaround = 0.115
    assert engine.prefix_length() == 2
    engine.turnarounds.append(0.18)
    assert engine.prefix_length() == 3
    engine.turnarounds.append(2.0)
    assert engine.prefix_length() == 8
    assert engine.refill_seconds == pytest.approx(2.0 + 1 / 15)
    engine.turnarounds.extend([0.115] * 100)
    assert engine.prefix_length() == 2


def test_dynamic_rules_match_lerobot_reference(config):
    # Load pure reference helpers without importing LeRobot/transformers 5 in the OpenPI environment.
    import ast
    import math
    from enum import Enum

    class Mode(Enum):
        RTC_TRAINED = 1

    source = (shared.OPENPI_ROOT.parents[1] / "inference/execution.py").read_text()
    nodes = [
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name in {"estimate_delay", "trained_overlap_valid"}
    ]
    namespace = {"math": math, "ExecutionMode": Mode}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<lerobot-reference>", "exec"), namespace)
    for available in [0, 1, 2, 8, 25]:
        for latency in [0.0, 0.05, 0.115, 0.18, 0.54, 2.0]:
            engine, _, _ = make_engine(config)
            engine.queue.extend(chunk()[:available])
            engine.initial_turnaround = latency
            assert engine.prefix_length() == namespace["estimate_delay"](
                latency, 1 / 15, Mode.RTC_TRAINED, 8, available
            )
    for prefix in [2, 3, 8]:
        for consumed in [0, 2, 4, 9]:
            engine, _, clock = make_engine(config)
            engine.prefix = chunk()[:prefix]
            engine.consumed = consumed
            engine.future = Future()
            engine.future.set_result(chunk())
            clock[0] = 0.115
            engine._accept()
            expected = namespace["trained_overlap_valid"](prefix, max(2, consumed), 8, True)
            assert (engine.inferences == 1) == expected


@pytest.mark.parametrize("steps", [1, 3, 25, 50])
def test_nonrtc_n_action_steps(config, steps):
    config.update(rtc=False, n_action_steps=steps)
    engine, ex, _ = make_engine(config)
    assert engine.tick({}, 0.0) is None
    ex.jobs[-1][0].set_result(chunk())
    for i in range(steps):
        assert engine.tick({}, 0.0)[0] == i
    assert engine.tick({}, 0.0) is None
    assert ex.jobs[-1][1][1].shape == (0, 16)


@pytest.mark.parametrize("result", [np.zeros((50, 15)), np.full((50, 16), np.nan)])
def test_reject_bad_model_output(config, result):
    engine, ex, _ = make_engine(config)
    engine.tick({}, 0.0)
    ex.jobs[-1][0].set_result(result)
    with pytest.raises(ValueError):
        engine.tick({}, 0.0)


def test_worker_failure_and_timeout_propagate(config):
    engine, ex, _ = make_engine(config)
    engine.tick({}, 0.0)
    ex.jobs[-1][0].set_exception(RuntimeError("inference failed"))
    with pytest.raises(RuntimeError, match="inference failed"):
        engine.tick({}, 0.0)
    engine, ex, clock = make_engine(config)
    engine.tick({}, 0.0)
    clock[0] = 6
    with pytest.raises(TimeoutError):
        engine.tick({}, 6.0)


def test_reject_prefix_changed(config):
    engine, ex, _ = make_engine(config)
    engine.tick({}, 0.0)
    ex.jobs[-1][0].set_result(chunk())
    for _ in range(26):
        engine.tick({}, 0.0)
    ex.jobs[-1][0].set_result(chunk(100))
    with pytest.raises(ValueError, match="prefix"):
        engine.tick({}, 0.0)


class FakeModel:
    def __init__(self):
        self.config = SimpleNamespace(action_horizon=5, action_dim=3)
        self.calls = []
        language = SimpleNamespace(config=SimpleNamespace())
        self.paligemma_with_expert = SimpleNamespace(
            paligemma=SimpleNamespace(language_model=language), forward=lambda **k: (None, "cache")
        )

    def _preprocess_observation(self, *a, **k):
        return None, None, None, None, torch.zeros(1, 3)

    def embed_prefix(self, *a):
        return torch.zeros(1, 2, 3), torch.ones(1, 2, dtype=torch.bool), torch.zeros(1, 2, dtype=torch.bool)

    def _prepare_attention_masks_4d(self, a):
        return a

    def denoise_step(self, state, pad, cache, x, t):
        self.calls.append((x.clone(), t.clone()))
        return torch.ones_like(x)


@pytest.mark.parametrize("count", [0, 2])
def test_sampler_clamps_clean_prefix_every_step(count):
    model = FakeModel()
    prefix = torch.full((1, count, 3), 7.0)
    result = sample_rtc(
        model, "cpu", SimpleNamespace(state=torch.zeros(1, 3)), prefix, 4, noise=torch.zeros(1, 5, 3)
    )
    assert len(model.calls) == 4
    for x, t in model.calls:
        if count:
            torch.testing.assert_close(x[:, :count], prefix)
            assert (t[:, :count] == 0).all()
            assert t.shape == (1, 5)
        else:
            assert t.shape == (1,)
    torch.testing.assert_close(result[:, :count], prefix)
    torch.testing.assert_close(result[:, count:], torch.full_like(result[:, count:], -1.0))


def test_menu_hides_incomplete_download(tmp_path):
    for name in ["037500", ".download-040000", "last", "040000.incomplete"]:
        p = tmp_path / name
        p.mkdir()
        for f in ["model.safetensors", "dataset_stats.json", "train_config.json"]:
            (p / f).touch()
    assert [p.name for p in checkpoints(tmp_path)] == ["037500"]


def test_real_checkpoint_contract(config):
    if not Path(config["checkpoint"]).is_dir():
        pytest.skip("No local checkpoint")
    result = inspect(config)
    assert result["tensors"] > 100
    bad = copy.deepcopy(config)
    bad["rtc_prefix_steps"] = 9
    with pytest.raises(ValueError, match="trained delay"):
        inspect(bad)


def test_no_motion_before_start_and_cleanup(config, tmp_path, monkeypatch):
    import flexiv_deploy_hardware

    import deploy

    calls = []

    class Robot:
        def __init__(self, *a, **k):
            pass

        def connect(self):
            calls.append("connect")

        def close(self):
            calls.append("close")

        def send_action(self, *a):
            raise AssertionError("Policy motion before start")

    class Log:
        def __init__(self, *a, **k):
            pass

        def close(self):
            calls.append("log_close")

    config.update(log_root=str(tmp_path), state_hz=0)
    monkeypatch.setattr(flexiv_deploy_hardware, "FlexivDualArm", Robot)
    monkeypatch.setattr(deploy, "DeploymentLog", Log)
    monkeypatch.setattr(deploy, "home", lambda *a: calls.append("home"))
    monkeypatch.setattr(deploy.time, "sleep", lambda *a: None)
    monkeypatch.setattr(deploy, "wait_for_continuous_start", lambda: False)
    deploy.execute(config, None, "continuous", (None, {"scene": "scene_0001"}))
    assert calls == ["connect", "home", "close", "log_close"]
    metadata = next(tmp_path.glob("*/metadata.json"))
    assert read_json(metadata)["checkpoint"] == config["checkpoint"]


@pytest.mark.parametrize("reject_plan", [False, True])
def test_home_float32_grippers_far_start_and_shared_sequence(config, monkeypatch, reject_plan):
    """Reproduce both real startup errors through the real hardware pose validator."""
    import flexiv_deploy_hardware
    import go_home_path

    import deploy

    events = []
    recorded = np.zeros(16, dtype=np.float32)
    recorded[[7, 15]] = 0.03
    original = recorded.copy()

    class Robot(flexiv_deploy_hardware.FlexivDualArm):
        def __init__(self):
            self.cfg = config["hardware"]
            self.start_pose = config["start_pose"]
            self._joint_min = np.full(14, -3.0)
            self._joint_max = np.full(14, 3.0)
            self.state = recorded.copy()
            self.state[1] = 1.2

        def read_state(self):
            return self.state.copy()

        def send_action(self, target, *, force_gripper_command=False):
            assert force_gripper_command
            target = flexiv_deploy_hardware.validate_target(
                target, self.state, self._joint_min, self._joint_max, self.cfg
            )
            events.append("open" if target[7] > 0 else "close")
            self.state = target.copy()
            return target

        def wait_for_target(self, target, timeout_s, **kwargs):
            events.append("opened")
            np.testing.assert_array_equal(target, self.state)

        def wait_for_grippers_closed(self, *, tolerance_m, timeout_s):
            events.append("closed")
            assert tolerance_m == 0.001
            np.testing.assert_array_equal(self.state[[7, 15]], [0.0, 0.0])
            return 0.0, 0.0

        def hold(self):
            events.append("hold")

    robot = Robot()
    # The two former failure conditions are present in this fixture.
    with pytest.raises(ValueError, match="Initial pose differs"):
        robot.check_initial_pose(recorded)
    bad_open = recorded.copy()
    bad_open[[7, 15]] = config["hardware"]["gripper_max_m"]
    with pytest.raises(ValueError, match="grippers exceed"):
        robot.check_initial_pose(bad_open, enforce_distance_limit=False)

    def plan(robot, cfg, target, tcp, lift_m, *, enforce_start_distance):
        events.append("preflight")
        assert enforce_start_distance is False
        assert target.dtype == np.float64
        assert target[7] == cfg["hardware"]["gripper_max_m"]
        assert lift_m == cfg["home_lift_m"]
        robot.check_initial_pose(target, enforce_distance_limit=enforce_start_distance)
        if reject_plan:
            raise ValueError("unreachable path")
        return object()

    def execute_path(robot, cfg, target, plan):
        events.append("lift_transfer_descend")
        robot.state = target.copy()

    monkeypatch.setattr(go_home_path, "read_home_tcp", lambda *args: np.zeros((2, 7)))
    monkeypatch.setattr(go_home_path, "plan_home", plan)
    monkeypatch.setattr(go_home_path, "execute_home", execute_path)
    if reject_plan:
        with pytest.raises(ValueError, match="unreachable path"):
            deploy.home(robot, config, (recorded, {}))
        assert events == ["preflight"]
    else:
        deploy.home(robot, config, (recorded, {}))
        assert events == ["preflight", "open", "opened", "lift_transfer_descend", "close", "closed", "hold"]
    np.testing.assert_array_equal(recorded, original)


def test_worker_completion_excludes_control_poll_delay(config):
    from engine import CompletedPrediction

    engine, ex, clock = make_engine(config)
    engine.tick({}, 0.0)
    # 115 ms inference; main thread only polls at 140 ms. Estimate must remain 2 steps.
    ex.jobs[-1][0].set_result(CompletedPrediction(chunk(), 0.115))
    clock[0] = 0.14
    engine.tick({}, 0.14)
    assert engine.last_inference_ms == pytest.approx(115.0)
    assert engine.prefix_length() == 2


def test_old_actions_expire_while_successor_pending(config):
    engine, ex, clock = make_engine(config)
    engine.queue.extend(chunk()[:10])
    engine.queue_capture = 0.0
    engine.has_executed = True
    clock[0] = 5.1
    with pytest.raises(TimeoutError, match="chunk observation expired"):
        engine.tick({}, 5.1)


@pytest.mark.parametrize("rtc", [True, False])
def test_near_home_skips_arm_return_for_both_deployment_modes(config, monkeypatch, rtc):
    import go_home_path

    import deploy

    config["rtc"] = rtc
    robot = MagicMock()
    recorded = np.zeros(16)
    measured = recorded.copy()
    measured[14] = 0.019
    robot.check_initial_pose.return_value = (measured, 0.019)
    monkeypatch.setattr(go_home_path, "read_home_tcp", lambda *args: np.zeros((2, 7)))
    plan = MagicMock(side_effect=AssertionError("Near-home startup must not plan an arm return"))
    monkeypatch.setattr(go_home_path, "plan_home", plan)
    deploy.home(robot, config, (recorded, {}))
    plan.assert_not_called()
    robot.send_action.assert_not_called()
    robot.wait_for_grippers_closed.assert_not_called()
    robot.hold.assert_called_once()


@pytest.fixture
def execution(config, tmp_path, monkeypatch):
    """Run the real deployment loop with a deterministic clock and no devices."""
    import flexiv_deploy_hardware
    import high_rate_state_recording

    import deploy

    runtime = SimpleNamespace(
        now=0.0, dispatch_times=[], events=[], observation_delays=deque(), tick_delays=deque()
    )
    robot = MagicMock()
    robot.camera_ages_s = dict.fromkeys(CAMERA_NAMES, 0.0)

    def observe():
        robot.observation_time = runtime.now
        if runtime.observation_delays:
            runtime.now += runtime.observation_delays.popleft()
        return dict.fromkeys(ACTION_NAMES, 0.0)

    def send(action):
        runtime.dispatch_times.append(runtime.now)
        return action.copy()

    def sleep(seconds):
        runtime.now += seconds

    robot.get_observation.side_effect = observe
    robot.send_action.side_effect = send
    robot.close.side_effect = lambda: runtime.events.append("stop_robot")
    engine = MagicMock()
    engine.queue = deque()
    for name in (
        "last_inference_ms",
        "inferences",
        "last_prefix_length",
        "discarded",
        "last_trimmed",
        "turnaround",
    ):
        setattr(engine, name, 0)

    def tick(*args):
        if runtime.tick_delays:
            runtime.now += runtime.tick_delays.popleft()
        return np.zeros(16)

    engine.tick.side_effect = tick
    engine.close.side_effect = lambda: runtime.events.append("stop_engine")
    log = MagicMock()
    log.close.side_effect = lambda: runtime.events.append("close_log")
    recorder = MagicMock()
    recorder.close.side_effect = lambda: runtime.events.append("close_recorder")

    def make_recorder(directory, cfg, hz):
        directory.mkdir(parents=True)
        (directory / "metadata.json").write_text("{}")
        return recorder

    config.update(log_root=str(tmp_path), state_hz=0, duration_s=0.5)
    monkeypatch.setattr(flexiv_deploy_hardware, "FlexivDualArm", lambda *a, **k: robot)
    monkeypatch.setattr(high_rate_state_recording, "StateRecorder", make_recorder)
    monkeypatch.setattr(deploy, "DeploymentLog", lambda *a, **k: log)
    monkeypatch.setattr(deploy, "ChunkEngine", lambda *a, **k: engine)
    monkeypatch.setattr(deploy, "home", lambda *a: None)
    monkeypatch.setattr(deploy, "wait_for_continuous_start", lambda: True)
    monkeypatch.setattr(deploy.time, "monotonic", lambda: runtime.now)
    monkeypatch.setattr(deploy.time, "sleep", sleep)
    policy = SimpleNamespace(predict=lambda *a: np.zeros((50, 16)))
    return SimpleNamespace(
        run=lambda mode="continuous": deploy.execute(
            config, policy, mode, (np.zeros(16), {"scene": "scene_0001"})
        ),
        runtime=runtime,
        robot=robot,
        engine=engine,
        log=log,
        recorder=recorder,
    )


@pytest.mark.parametrize("phase", ["observation", "inference"])
def test_control_overrun_does_not_catch_up_with_adjacent_commands(config, execution, phase):
    delays = execution.runtime.observation_delays if phase == "observation" else execution.runtime.tick_delays
    delays.append(0.2)
    execution.run()
    dispatched = execution.runtime.dispatch_times
    assert len(dispatched) >= 2
    assert np.all(np.diff(dispatched) >= 1 / config["fps"] - 1e-9)


def test_single_step_deadline_includes_post_inference_camera_check(config, execution, monkeypatch):
    answers = iter(["", "q"])
    monkeypatch.setattr("builtins.input", lambda *a: next(answers))
    execution.runtime.observation_delays.extend([0, config["max_observation_age_s"] + 0.1])
    with pytest.raises(TimeoutError, match="Single-step inference expired"):
        execution.run("step")
    execution.robot.send_action.assert_not_called()
    execution.robot.close.assert_called_once()


def test_state_recorder_close_failure_still_closes_deployment_log(config, execution):
    config.update(state_hz=200, duration_s=0.05)

    def fail_close():
        execution.runtime.events.append("close_recorder")
        raise OSError("disk close failed")

    execution.recorder.close.side_effect = fail_close
    with pytest.raises(OSError, match="disk close failed"):
        execution.run()
    assert execution.runtime.events == ["stop_robot", "stop_engine", "close_recorder", "close_log"]
