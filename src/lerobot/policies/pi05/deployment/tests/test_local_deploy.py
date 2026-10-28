"""Pure/mocked checks only. No policy weights, SDK imports, USB, or robot connections."""

# RDK's public method names are preserved by the fake SDK below.
# ruff: noqa: N802

from __future__ import annotations

import copy
import importlib.util
import io
import json
import struct
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import deploy_dualarm
import deploy_dualarm_no_rtc
import numpy as np
from deploy_dualarm import (
    align_start_pose,
    available_checkpoint_folders,
    available_checkpoints,
    choose_checkpoint,
    choose_run_mode,
    dispatch_single_step,
    execute as execute_deployment,
    main as deploy_main,
    read_start_pose,
    require_compatible_rdk,
)
from flexiv_deploy_contract import (
    ACTION_NAMES,
    inspect_checkpoint,
    read_json,
    tensor_header,
    validate_settings,
)
from flexiv_deploy_hardware import CameraFreshnessError, FlexivDualArm, letterbox, pack_state, validate_target
from goHome import main as go_home_main
from init_grippers import main as init_grippers_main


def write_tensors(path: Path, names: list[str]) -> None:
    header = {
        name: {"dtype": "F32", "shape": [16], "data_offsets": [i * 64, (i + 1) * 64]}
        for i, name in enumerate(names)
    }
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(64 * len(names)))


def mock_configuration() -> dict:
    """Keep mocked controller tests independent of user-selected live controller settings."""
    cfg = read_json(Path(__file__).resolve().parents[1] / "configs/deploy_config.json")
    cfg["hardware"]["control_mode"] = "NRT_JOINT_POSITION"
    cfg["hardware"]["max_target_error_rad"] = 0.25
    for side in ("left", "right"):
        cfg["hardware"]["arms"][side]["max_contact_torque_nm"] = None
    return cfg


class TestContract(unittest.TestCase):
    def setUp(self):
        self.cfg = mock_configuration()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cfg["checkpoint"] = str(self.root)
        self.cfg["base_model"] = str(self.root / "base")
        self.cfg["tokenizer"] = str(self.root / "tokenizer")
        (self.root / "tokenizer").mkdir()
        for filename in ("tokenizer_config.json", "tokenizer.json"):
            (self.root / "tokenizer" / filename).write_text("{}")
        self.model = {
            "type": "pi05",
            "chunk_size": 50,
            "rtc_training_max_delay": 8,
            "use_relative_actions": True,
            "relative_exclude_joints": ["gripper"],
            "action_feature_names": ACTION_NAMES.copy(),
            "use_peft": False,
            "input_features": {
                "observation.state": {"shape": [16]},
                **{
                    f"observation.images.{name}": {"shape": [3, 224, 224]}
                    for name in ("image", "image2", "image3")
                },
            },
            "output_features": {"action": {"shape": [16]}},
        }
        self._save_model()
        pre = {
            "steps": [
                {
                    "registry_name": "relative_actions_processor",
                    "config": {
                        "enabled": True,
                        "exclude_joints": ["gripper"],
                        "action_names": ACTION_NAMES.copy(),
                    },
                },
                {"registry_name": "normalizer_processor", "state_file": "pre.safetensors"},
            ]
        }
        post = {
            "steps": [
                {"registry_name": "unnormalizer_processor", "state_file": "post.safetensors"},
                {"registry_name": "absolute_actions_processor", "config": {"enabled": True}},
            ]
        }
        (self.root / "policy_preprocessor.json").write_text(json.dumps(pre))
        (self.root / "policy_postprocessor.json").write_text(json.dumps(post))
        write_tensors(
            self.root / "pre.safetensors",
            ["observation.state.q01", "observation.state.q99", "action.q01", "action.q99"],
        )
        write_tensors(self.root / "post.safetensors", ["action.q01", "action.q99"])
        write_tensors(self.root / "model.safetensors", ["test_weight"])

    def _save_model(self):
        (self.root / "config.json").write_text(json.dumps(self.model))

    def test_complete_full_checkpoint_and_distinct_follower_mapping(self):
        self.assertEqual(inspect_checkpoint(self.cfg)["kind"], "full")
        self.assertEqual(self.cfg["hardware"]["arms"]["left"]["serial"], "Rizon4-063906")
        self.assertEqual(self.cfg["hardware"]["arms"]["right"]["serial"], "Rizon4R-062106")

    def test_lora_requires_local_base_and_complete_adapter(self):
        self.model["use_peft"] = True
        self._save_model()
        (self.root / "adapter_config.json").write_text("{}")
        write_tensors(self.root / "adapter_model.safetensors", ["test_lora"])
        with self.assertRaises(FileNotFoundError):
            inspect_checkpoint(self.cfg)
        (self.root / "base").mkdir()
        write_tensors(self.root / "base/model.safetensors", ["test_base"])
        self.assertEqual(inspect_checkpoint(self.cfg)["kind"], "LoRA")

    def test_missing_statistics_and_truncated_weights_fail(self):
        (self.root / "pre.safetensors").unlink()
        with self.assertRaises(FileNotFoundError):
            inspect_checkpoint(self.cfg)
        path = self.root / "model.safetensors"
        path.write_bytes(path.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError, "incomplete"):
            tensor_header(path)

    def test_permuted_joint_order_or_mismatched_trained_rtc_is_rejected(self) -> None:
        self.model["action_feature_names"][0], self.model["action_feature_names"][8] = (
            self.model["action_feature_names"][8],
            self.model["action_feature_names"][0],
        )
        self._save_model()
        with self.assertRaisesRegex(ValueError, "order"):
            inspect_checkpoint(self.cfg)
        self.model["action_feature_names"] = ACTION_NAMES.copy()
        self.model["rtc_training_max_delay"] = 4
        self._save_model()
        with self.assertRaisesRegex(ValueError, "rtc_training_max_delay"):
            inspect_checkpoint(self.cfg)

    def test_non_rtc_accepts_both_training_types_but_rtc_requires_trained_weights(self) -> None:
        self.assertEqual(inspect_checkpoint(self.cfg)["kind"], "full")
        before = (self.root / "config.json").read_bytes()
        self.assertEqual(inspect_checkpoint(self.cfg, rtc=False)["kind"], "full")
        self.assertEqual((self.root / "config.json").read_bytes(), before)
        for name in ("training_max_delay", "rtc_execution_horizon", "rtc_queue_threshold"):
            no_rtc_cfg = {key: value for key, value in self.cfg.items() if key != name}
            self.assertEqual(inspect_checkpoint(no_rtc_cfg, rtc=False)["kind"], "full")
        self.model["rtc_training_max_delay"] = 0
        self._save_model()
        self.cfg["inference_mode"] = "chunk"
        with self.assertRaisesRegex(ValueError, "rtc_training_max_delay"):
            inspect_checkpoint(self.cfg)
        self.assertEqual(inspect_checkpoint(self.cfg, rtc=False)["kind"], "full")
        for name in ("training_max_delay", "rtc_execution_horizon", "rtc_queue_threshold"):
            del self.cfg[name]
        self.assertEqual(inspect_checkpoint(self.cfg, rtc=False)["kind"], "full")

    def test_invalid_delay_and_chunk_execution_length_are_rejected(self) -> None:
        for value in (-1, True, 0.5):
            with self.subTest(delay=value):
                self.model["rtc_training_max_delay"] = value
                self._save_model()
                with self.assertRaisesRegex(ValueError, "rtc_training_max_delay"):
                    inspect_checkpoint(self.cfg)
                with self.assertRaisesRegex(ValueError, "rtc_training_max_delay"):
                    inspect_checkpoint(self.cfg, rtc=False)
        self.model["rtc_training_max_delay"] = 0
        for value in (0, 51, True, 2.5):
            with self.subTest(execution_steps=value):
                self.model["n_action_steps"] = value
                self._save_model()
                with self.assertRaisesRegex(ValueError, "n_action_steps"):
                    inspect_checkpoint(self.cfg, rtc=False)

    def test_non_rtc_deployment_execution_steps_must_fit_the_predicted_chunk(self) -> None:
        self.model["rtc_training_max_delay"] = 0
        self._save_model()
        for value in (1, 5, 50):
            with self.subTest(execution_steps=value):
                self.cfg["n_action_steps"] = value
                self.assertEqual(inspect_checkpoint(self.cfg, rtc=False)["kind"], "full")
        for invalid_steps in (-1, 0, 51, True, 2.5, None):
            with self.subTest(execution_steps=invalid_steps):
                self.cfg["n_action_steps"] = invalid_steps
                with self.assertRaisesRegex(ValueError, "n_action_steps must be an integer between 1 and 50"):
                    inspect_checkpoint(self.cfg, rtc=False)

    def test_invalid_timing_and_duplicate_serials_are_rejected(self):
        for field, value in (("fps", 30), ("rtc_execution_horizon", 7), ("rtc_queue_threshold", 4)):
            cfg = copy.deepcopy(self.cfg)
            cfg[field] = value
            with self.assertRaises(ValueError):
                validate_settings(cfg)
        cfg = copy.deepcopy(self.cfg)
        cfg["start_pose"]["max_joint_delta_rad"] = 1.01
        with self.assertRaisesRegex(ValueError, "safety bounds"):
            validate_settings(cfg)
        self.cfg["hardware"]["arms"]["right"]["serial"] = self.cfg["hardware"]["arms"]["left"]["serial"]
        with self.assertRaisesRegex(ValueError, "distinct"):
            validate_settings(self.cfg)

    def test_contact_torque_requires_impedance_mode_and_valid_seven_joint_values(self):
        hw = self.cfg["hardware"]
        hw["arms"]["right"]["max_contact_torque_nm"] = [1.0] * 7
        with self.assertRaisesRegex(ValueError, "right.*NRT_JOINT_IMPEDANCE"):
            validate_settings(self.cfg)
        hw["control_mode"] = "NRT_JOINT_IMPEDANCE"
        validate_settings(self.cfg)
        for invalid in (
            [],
            [1] * 6,
            [1] * 8,
            10,
            "10",
            [-1] * 7,
            [True] * 7,
            [float("nan")] * 7,
            [float("inf")] * 7,
            ["1"] * 7,
        ):
            with self.subTest(limits=invalid):
                hw["arms"]["right"]["max_contact_torque_nm"] = invalid
                with self.assertRaisesRegex(ValueError, "right.*seven finite"):
                    validate_settings(self.cfg)


class TestCheckpointMenu(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.rtc_root = self.root / "rtc_full"
        self.no_rtc_root = self.root / "DualArm_FoldTowel_pi05_no_rtc_full_gpu7_6_5_4"
        for step in ("030000", "040000", "045000", "050000"):
            checkpoint = self.rtc_root / step / "pretrained_model"
            checkpoint.mkdir(parents=True)
            (checkpoint / "config.json").write_text("{}")
            (checkpoint / "model.safetensors").write_bytes(b"weight")
        (self.rtc_root / "last").symlink_to("050000", target_is_directory=True)
        checkpoint = self.no_rtc_root / "025000" / "pretrained_model"
        checkpoint.mkdir(parents=True)
        (checkpoint / "config.json").write_text("{}")
        (checkpoint / "model.safetensors").write_bytes(b"weight")
        (checkpoint / "tokenizer").mkdir()

    def test_lists_rtc_full_checkpoints_newest_first_without_duplicate_last_link(self) -> None:
        choices = available_checkpoints(self.rtc_root)
        self.assertEqual(
            [(item[0], item[1], item[3]) for item in choices],
            [
                ("rtc_full", 50000, "full"),
                ("rtc_full", 45000, "full"),
                ("rtc_full", 40000, "full"),
                ("rtc_full", 30000, "full"),
            ],
        )
        with patch("builtins.input", side_effect=["0", "2", "0", "3"]), redirect_stdout(io.StringIO()):
            selected = choose_checkpoint(self.root)
        self.assertEqual(selected, choices[2][2])

    def test_lists_only_folders_with_weights(self) -> None:
        (self.root / "tokenizer").mkdir()
        for name in ("incomplete", ".download"):
            checkpoint = self.root / name / "060000" / "pretrained_model"
            checkpoint.mkdir(parents=True)
            (checkpoint / "config.json").write_text("{}")
        (self.root / ".download/060000/pretrained_model/model.safetensors").write_bytes(b"weight")
        self.assertEqual(available_checkpoint_folders(self.root), [self.no_rtc_root, self.rtc_root])

    def test_checkpoint_menu_only_lists_the_selected_folder(self) -> None:
        output = io.StringIO()
        with patch("builtins.input", side_effect=["1", "2", "1"]) as keys, redirect_stdout(output):
            selected = choose_checkpoint(self.root)
        self.assertEqual(selected, self.no_rtc_root / "025000/pretrained_model")
        self.assertEqual(keys.call_count, 3)
        self.assertIn("step 25,000", output.getvalue())
        self.assertNotIn("step 50,000", output.getvalue())

    def test_training_output_is_discovered_in_rtc_and_non_rtc_menus(self) -> None:
        experiment = self.root / "DualArm_FoldTowel_pi05_rtc_full_finetune"
        for step in ("010000", "050000"):
            model = experiment / "checkpoints" / step / "pretrained_model"
            model.mkdir(parents=True)
            (model / "config.json").write_text("{}")
            (model / "model.safetensors").write_bytes(b"weight")
        (experiment / "checkpoints/last").symlink_to("050000", target_is_directory=True)
        for module in (deploy_dualarm, deploy_dualarm_no_rtc):
            with self.subTest(entry=module.__name__):
                choices = module.available_checkpoints(experiment)
                self.assertEqual(
                    [(item[0], item[1]) for item in choices],
                    [(experiment.name, 50000), (experiment.name, 10000)],
                )
                folders = module.available_checkpoint_folders(self.root)
                index = folders.index(experiment) + 1
                with patch("builtins.input", side_effect=[str(index), "1"]), redirect_stdout(io.StringIO()):
                    selected = module.choose_checkpoint(self.root)
                self.assertEqual(selected, experiment / "checkpoints/050000/pretrained_model")
                with patch("builtins.input", return_value="1") as keys, redirect_stdout(io.StringIO()):
                    selected = module.choose_checkpoint(experiment)
                self.assertEqual(keys.call_count, 1)
                self.assertEqual(selected, experiment / "checkpoints/050000/pretrained_model")

    def test_legacy_config_can_still_select_from_one_experiment_folder(self) -> None:
        with patch("builtins.input", return_value="2"), redirect_stdout(io.StringIO()):
            selected = choose_checkpoint(self.rtc_root)
        self.assertEqual(selected, self.rtc_root / "045000/pretrained_model")

    def test_menu_reads_vision_training_status_from_saved_config(self) -> None:
        for step, frozen in (("050000", True), ("045000", False)):
            (self.rtc_root / step / "pretrained_model/config.json").write_text(
                json.dumps({"freeze_vision_encoder": frozen})
            )
        output = io.StringIO()
        with patch("builtins.input", return_value="q"), redirect_stdout(output):
            choose_checkpoint(self.rtc_root)
        lines = output.getvalue().splitlines()
        self.assertTrue(any("step 50,000" in line and "视觉冻结" in line for line in lines))
        self.assertTrue(any("step 45,000" in line and "视觉参与训练" in line for line in lines))

    def test_empty_root_reports_no_checkpoints(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "No local checkpoints"):
            choose_checkpoint(self.root / "missing")

    def test_cancel_at_either_menu_exits_before_loading_or_connecting(self) -> None:
        cfg = {"checkpoint": str(self.root / "pretrained_model"), "checkpoint_root": str(self.root)}
        for answers in (["q"], ["2", "q"]):
            with (
                self.subTest(answers=answers),
                patch.object(sys, "argv", ["deploy_dualarm.py"]),
                patch("deploy_dualarm.read_json", return_value=cfg),
                patch("builtins.input", side_effect=answers),
                patch("deploy_dualarm.inspect_checkpoint") as inspect,
                patch("deploy_dualarm.require_compatible_rdk") as require_rdk,
                patch("deploy_dualarm.load_policy") as load_policy,
                patch("deploy_dualarm.execute") as execute,
                redirect_stdout(io.StringIO()),
            ):
                deploy_main()
            inspect.assert_not_called()
            require_rdk.assert_not_called()
            load_policy.assert_not_called()
            execute.assert_not_called()

    def test_eof_at_either_menu_exits_before_loading(self) -> None:
        for answers in ([EOFError], ["2", EOFError]):
            with (
                self.subTest(answers=answers),
                patch("builtins.input", side_effect=answers),
                redirect_stdout(io.StringIO()),
                self.assertRaisesRegex(SystemExit, "--checkpoint"),
            ):
                choose_checkpoint(self.root)

    def test_selected_checkpoint_reaches_execution_without_using_default_link(self) -> None:
        cfg = read_json(Path(__file__).resolve().parents[1] / "configs/deploy_config.json")
        cfg["checkpoint"] = str(self.root / "pretrained_model")
        cfg["checkpoint_root"] = str(self.root)
        expected = self.rtc_root / "050000/pretrained_model"
        with (
            patch.dict(sys.modules, {"flexivrdk": SimpleNamespace(__version__="1.9.2")}),
            patch.object(sys, "argv", ["deploy_dualarm.py"]),
            patch("deploy_dualarm.read_json", return_value=cfg),
            patch("builtins.input", side_effect=["2", "1", "1"]),
            patch("deploy_dualarm.inspect_checkpoint", return_value={"kind": "full"}),
            patch("deploy_dualarm.read_start_pose", return_value=(np.zeros(16), {})),
            patch("deploy_dualarm.load_policy", return_value=(None, None, None)) as load_policy,
            patch("deploy_dualarm.execute") as execute,
            redirect_stdout(io.StringIO()),
        ):
            deploy_main()
        self.assertEqual(Path(load_policy.call_args.args[0]["checkpoint"]), expected)
        execute.assert_called_once()

    def test_explicit_checkpoint_skips_both_checkpoint_menus(self) -> None:
        expected = self.rtc_root / "040000/pretrained_model"
        cfg = read_json(Path(__file__).resolve().parents[1] / "configs/deploy_config.json")
        with (
            patch.object(sys, "argv", ["deploy_dualarm.py", "--checkpoint", str(expected), "--check"]),
            patch("deploy_dualarm.read_json", return_value=cfg),
            patch("builtins.input", side_effect=AssertionError("unexpected menu")),
            patch("deploy_dualarm.inspect_checkpoint", return_value={"kind": "full"}) as inspect,
            patch("deploy_dualarm.load_policy") as load_policy,
            patch("deploy_dualarm.execute") as execute,
        ):
            deploy_main()
        self.assertEqual(Path(inspect.call_args.args[0]["checkpoint"]), expected)
        load_policy.assert_not_called()
        execute.assert_not_called()

    def test_incompatible_rdk_fails_before_hardware_execution(self):
        with (
            patch.dict(sys.modules, {"flexivrdk": SimpleNamespace(__version__="1.9.3")}),
            self.assertRaisesRegex(RuntimeError, "requires Flexiv RDK 1.9.2"),
        ):
            require_compatible_rdk()

    def test_check_only_does_not_load_model_or_connect_devices(self):
        cfg = read_json(Path(__file__).resolve().parents[1] / "configs/deploy_config.json")
        with (
            patch.object(sys, "argv", ["deploy_dualarm.py", "--check"]),
            patch("deploy_dualarm.read_json", return_value=cfg),
            patch("deploy_dualarm.inspect_checkpoint", return_value={"kind": "LoRA"}),
            patch("deploy_dualarm.load_policy") as load_policy,
            patch("deploy_dualarm.execute") as execute,
        ):
            deploy_main()
        load_policy.assert_not_called()
        execute.assert_not_called()

    def test_run_mode_choice_and_cancel(self):
        with patch("builtins.input", side_effect=["3", "1"]), redirect_stdout(io.StringIO()):
            self.assertEqual(choose_run_mode(), "step")
        with patch("builtins.input", return_value="q"), redirect_stdout(io.StringIO()):
            self.assertIsNone(choose_run_mode())


@unittest.skipUnless(importlib.util.find_spec("h5py"), "h5py required for source HDF5")
class TestStartPoseSource(unittest.TestCase):
    def test_reads_first_retained_frame_with_converter_interpolation_and_order(self):
        import h5py

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "scene_0001/lowdim/lowdim.h5"
            source.parent.mkdir(parents=True)
            with h5py.File(source, "w") as lowdim:
                lowdim["timestamp"] = [0, 100, 100, 200]
                for serial, sign in (("063906", 1), ("062106", -1)):
                    lowdim[f"joint_position_rad_{serial}"] = np.array(
                        [[value * sign] * 7 for value in (0, 1, 2, 3)], dtype=np.float32
                    )
                    lowdim[f"ee_state_{serial}"] = np.array(
                        [[value] for value in (0, 0.02, 0.04, 0.06)], dtype=np.float32
                    )
            manifest = {
                "source": str(root),
                "task": "DualArm_FoldTowel",
                "fps": 15,
                "max_lowdim_gap_ms": 110,
                "dimension_names": ACTION_NAMES,
                "arms": {"left": "063906", "right": "062106"},
                "scenes": [{"scene": "scene_0001", "segments": [{"source_start_ms": 150}]}],
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            cfg = read_json(Path(__file__).resolve().parents[1] / "configs/deploy_config.json")
            cfg["start_pose"]["manifest"] = str(manifest_path)
            state, info = read_start_pose(cfg)
            np.testing.assert_allclose(state[:7], [2.5] * 7)
            np.testing.assert_allclose(state[8:15], [-2.5] * 7)
            np.testing.assert_allclose(state[[7, 15]], [0.05, 0.05])
            self.assertEqual(info["timestamp_ms"], 150)
            manifest["max_lowdim_gap_ms"] = 50
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "sampling gap"):
                read_start_pose(cfg)


class TestStepDispatch(unittest.TestCase):
    def test_alignment_starts_without_prompt_after_pose_check(self):
        robot = MagicMock()
        robot.check_initial_pose.return_value = (np.zeros(16), 0.0)
        robot.move_to_initial_pose.return_value = np.zeros(16)
        start_pose = (np.zeros(16), {"scene": "scene_0001", "timestamp_ms": 150.0})
        with (
            patch("builtins.input", side_effect=AssertionError("unexpected prompt")),
            self.assertLogs(level="INFO"),
        ):
            align_start_pose(robot, start_pose)
        robot.check_initial_pose.assert_called_once()
        robot.move_to_initial_pose.assert_called_once()

    def test_alignment_rejects_bad_pose_before_motion(self):
        robot = MagicMock()
        robot.check_initial_pose.side_effect = ValueError("too far")
        start_pose = (np.zeros(16), {"scene": "scene_0001", "timestamp_ms": 150.0})
        with self.assertRaisesRegex(ValueError, "too far"):
            align_start_pose(robot, start_pose, close_grippers=True)
        robot.send_action.assert_not_called()
        robot.move_to_initial_pose.assert_not_called()

    def test_go_home_commands_closed_grippers_before_arm_alignment(self):
        robot = MagicMock()
        robot.check_initial_pose.return_value = (np.zeros(16), 0.0)
        robot.move_to_initial_pose.return_value = np.zeros(16)
        order = []
        robot.send_action.side_effect = lambda _, **kwargs: order.append("close")
        robot.move_to_initial_pose.side_effect = lambda _: order.append("align") or np.zeros(16)
        start_pose = (np.zeros(16), {"scene": "scene_0001", "timestamp_ms": 150.0})
        align_start_pose(robot, start_pose, close_grippers=True)
        self.assertEqual(order, ["close", "align"])
        np.testing.assert_array_equal(robot.send_action.call_args.args[0][[7, 15]], [0.0, 0.0])
        self.assertTrue(robot.send_action.call_args.kwargs["force_gripper_command"])

    def test_one_request_dispatches_one_action_and_pauses_rtc(self):
        class Action:
            def detach(self):
                return self

            def float(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return np.ones(16)

        engine, robot = MagicMock(), MagicMock()
        engine.failed = False
        engine.dispatch_allowed.return_value = True
        engine.get_action.return_value = Action()
        robot.get_observation.return_value = {"observation.state": np.zeros(16)}
        robot.send_action.return_value = np.ones(16)
        robot.observation_time = 123.0
        cfg = {"action_timeout_s": 1.0, "fps": 15}
        _, target, captured_at = dispatch_single_step(engine, robot, cfg)
        np.testing.assert_array_equal(target, np.ones(16))
        self.assertEqual(captured_at, 123.0)
        robot.send_action.assert_called_once()
        engine.reset.assert_called_once()
        engine.pause.assert_called_once()


class TestGoHome(unittest.TestCase):
    def setUp(self) -> None:
        self.enterContext(patch("goHome.read_home_tcp", return_value=np.zeros((2, 7))))
        self.enterContext(patch("go_home_path.plan_home"))

    def test_check_only_reads_first_frame_without_connecting(self):
        start_pose = (np.zeros(16), {"scene": "scene_0001", "timestamp_ms": 150.0})
        with (
            patch.object(sys, "argv", ["goHome.py", "--check"]),
            patch("goHome.read_start_pose", return_value=start_pose),
            patch("goHome.require_compatible_rdk") as check_rdk,
            patch("goHome.FlexivDualArm") as robot_class,
        ):
            go_home_main()
        check_rdk.assert_not_called()
        robot_class.assert_not_called()

    def test_uses_arm_only_connection_and_closes_after_alignment(self):
        recorded = np.zeros(16)
        recorded[[7, 15]] = 0.05
        start_pose = (recorded, {"scene": "scene_0001", "timestamp_ms": 150.0})
        with (
            patch.object(sys, "argv", ["goHome.py"]),
            patch("goHome.read_start_pose", return_value=start_pose),
            patch("goHome.require_compatible_rdk"),
            patch("goHome.FlexivDualArm") as robot_class,
            patch("go_home_path.execute_home") as align,
        ):
            robot = robot_class.return_value
            robot.check_initial_pose.return_value = (np.zeros(16), 0.0)
            robot.read_state.return_value = np.zeros(16)
            robot.wait_for_grippers_closed.return_value = (0.0, 0.0)
            events = []
            robot.send_action.side_effect = lambda target, **kwargs: (
                events.append(("gripper", target.copy())) or target
            )
            robot.wait_for_target.side_effect = lambda *args, **kwargs: events.append(("opened", None))
            align.side_effect = lambda *args: events.append(("align", None))
            go_home_main()
        robot = robot_class.return_value
        robot.connect.assert_called_once_with(with_cameras=False)
        self.assertIs(align.call_args.args[0], robot)
        np.testing.assert_allclose(align.call_args.args[2][[7, 15]], [0.1, 0.1])
        np.testing.assert_array_equal(recorded[[7, 15]], [0.05, 0.05])
        self.assertEqual([event[0] for event in events], ["gripper", "opened", "align", "gripper"])
        np.testing.assert_allclose(events[0][1][[7, 15]], [0.1, 0.1])
        robot.initialize_grippers.assert_not_called()
        np.testing.assert_allclose(events[-1][1][[7, 15]], [0.0, 0.0])
        self.assertTrue(
            all(call.kwargs["force_gripper_command"] for call in robot.send_action.call_args_list)
        )
        robot.wait_for_grippers_closed.assert_called_once_with(tolerance_m=0.001, timeout_s=10.0)
        robot.close.assert_called_once_with()

    def test_opening_timeout_prevents_arm_alignment_and_still_cleans_up(self):
        with (
            patch.object(sys, "argv", ["goHome.py"]),
            patch(
                "goHome.read_start_pose",
                return_value=(np.zeros(16), {"scene": "scene_0001", "timestamp_ms": 0}),
            ),
            patch("goHome.require_compatible_rdk"),
            patch("goHome.FlexivDualArm") as robot_class,
            patch("go_home_path.execute_home") as align,
        ):
            robot = robot_class.return_value
            robot.check_initial_pose.return_value = (np.zeros(16), 0.0)
            robot.wait_for_target.side_effect = TimeoutError("grippers did not open")
            with self.assertRaisesRegex(TimeoutError, "did not open"):
                go_home_main()
        align.assert_not_called()
        robot.send_action.assert_called_once()
        robot.close.assert_called_once()


class TestInitGrippers(unittest.TestCase):
    def test_check_does_not_connect_or_require_recording(self) -> None:
        with (
            patch.object(sys, "argv", ["init_grippers.py", "--check"]),
            patch("init_grippers.require_compatible_rdk") as check_rdk,
            patch("init_grippers.FlexivDualArm") as robot_class,
            patch("deploy_dualarm.read_start_pose", side_effect=AssertionError("unexpected recording read")),
        ):
            init_grippers_main()
        robot_class.assert_not_called()
        check_rdk.assert_not_called()

    def test_init_only_and_cleanup_on_success_or_failure(self) -> None:
        for error in (None, TimeoutError("Init timeout"), KeyboardInterrupt()):
            with (
                self.subTest(error=error),
                patch.object(sys, "argv", ["init_grippers.py"]),
                patch("init_grippers.require_compatible_rdk"),
                patch("init_grippers.FlexivDualArm") as robot_class,
                patch(
                    "deploy_dualarm.read_start_pose", side_effect=AssertionError("unexpected recording read")
                ),
            ):
                robot = robot_class.return_value
                robot.initialize_grippers.side_effect = error
                if error is None:
                    init_grippers_main()
                else:
                    with self.assertRaises(type(error)):
                        init_grippers_main()
                self.assertEqual(
                    [call[0] for call in robot.method_calls], ["connect", "initialize_grippers", "close"]
                )
                robot.connect.assert_called_once_with(with_cameras=False)


class TestTransforms(unittest.TestCase):
    def test_training_order_preserves_radians_and_meter_gripper(self):
        state = pack_state(np.full(7, 0.5), 0.03, np.full(7, -0.5), 0.08)
        self.assertEqual(state.dtype, np.float32)
        np.testing.assert_allclose(state, [0.5] * 7 + [0.03] + [-0.5] * 7 + [0.08])
        with self.assertRaises(ValueError):
            pack_state(np.zeros(8), 0.03, np.zeros(7), 0.08)

    def test_rgb_letterbox_matches_converter_without_channel_swap(self):
        rgb = np.zeros((720, 1280, 3), dtype=np.uint8)
        rgb[:, :, 0] = 255
        image = letterbox(rgb)
        self.assertEqual(image.shape, (224, 224, 3))
        np.testing.assert_array_equal(image[0], 0)
        np.testing.assert_array_equal(image[112, 112], [255, 0, 0])

    def test_targets_are_absolute_and_bad_targets_fail_before_dispatch(self):
        cfg = mock_configuration()["hardware"]
        measured = pack_state(np.full(7, 0.5), 0.03, np.full(7, -0.5), 0.08)
        action = measured.copy()
        action[0], action[7] = 0.6, 0.12
        target = validate_target(action, measured, np.full(14, -3), np.full(14, 3), cfg)
        self.assertAlmostEqual(target[0], 0.6, places=6)  # Not 0.5 + 0.6.
        self.assertEqual(target[7], 0.1)
        for bad in (np.full(16, np.nan), np.zeros(32), np.full(16, 20)):
            with self.assertRaises(ValueError):
                validate_target(bad, measured, np.full(14, -3), np.full(14, 3), cfg)

    def test_tracking_error_identifies_both_arms_and_configured_limit(self) -> None:
        cfg = mock_configuration()["hardware"]
        cfg["max_target_error_rad"] = 0.1
        measured = pack_state(np.full(7, 0.5), 0.03, np.full(7, -0.5), 0.08)
        action = measured.copy()
        action[1] += 0.12
        action[13] -= 0.2
        with self.assertRaises(ValueError) as caught:
            validate_target(action, measured, np.full(14, -3), np.full(14, 3), cfg)
        message = str(caught.exception)
        self.assertIn("max_error=0.20000 rad (limit=0.10000 rad)", message)
        self.assertIn("left A2: measured=0.50000, target=0.62000, error=0.12000 rad", message)
        self.assertIn("right A6: measured=-0.50000, target=-0.70000, error=0.20000 rad", message)
        self.assertIn("target rejected before dispatch", message)


class TestExecutionInitialization(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = read_json(Path(__file__).resolve().parents[1] / "configs/deploy_config.json")
        self.cfg["duration_s"] = 0.0
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.cfg["telemetry"] = str(Path(directory.name) / "telemetry.jsonl")
        self.order: list[str | tuple[str, float]] = []
        self.robot, self.engine = MagicMock(), MagicMock()
        self.robot.connect.side_effect = lambda: self.order.append("connect")
        self.robot.hold.side_effect = lambda: self.order.append("hold")
        self.robot.wait_for_fresh_observation.side_effect = lambda **kwargs: self.order.append("fresh_frames")
        self.engine.start.side_effect = lambda: self.order.append("start")
        self.engine.resume.side_effect = lambda: self.order.append("resume")

        def make_engine(**kwargs: object) -> MagicMock:
            self.order.append("construct_engine")
            return self.engine

        self.make_rtc_engine = MagicMock(side_effect=make_engine)
        self.enterContext(
            patch.dict(
                sys.modules,
                {
                    "lerobot.inference": SimpleNamespace(RTCInferenceEngine=self.make_rtc_engine),
                    "lerobot.policies.rtc.configuration_rtc": SimpleNamespace(RTCConfig=MagicMock()),
                },
            )
        )
        self.enterContext(patch("flexiv_deploy_hardware.FlexivDualArm", return_value=self.robot))
        self.align = self.enterContext(
            patch(
                "deploy_dualarm.align_deployment_start_pose",
                side_effect=lambda *_: self.order.append("align"),
            )
        )
        self.enterContext(
            patch(
                "deploy_dualarm.time.sleep",
                side_effect=lambda seconds: self.order.append(("settle", seconds)),
            )
        )

    def _run(self, mode: str = "continuous") -> None:
        execute_deployment(self.cfg, None, None, None, mode, (np.zeros(16), {"scene": "scene_0001"}))

    def _assert_cancelled_at_home(self) -> None:
        self.assertIn("align", self.order)
        self.robot.hold.assert_called_once()
        self.robot.send_action.assert_not_called()
        self.robot.wait_for_fresh_observation.assert_not_called()
        self.engine.start.assert_not_called()
        self.engine.resume.assert_not_called()
        self.engine.notify_observation.assert_not_called()
        self.robot.close.assert_called_once()
        self.engine.stop.assert_called_once()

    def test_home_and_start_key_precede_policy_execution(self) -> None:
        def press_start(prompt: str) -> str:
            self.assertIn("home", prompt)
            self.engine.start.assert_not_called()
            self.engine.resume.assert_not_called()
            self.robot.send_action.assert_not_called()
            self.order.append("start_key")
            return ""

        with patch("builtins.input", side_effect=press_start):
            self._run()
        self.assertEqual(
            self.order,
            [
                "construct_engine",
                "connect",
                "align",
                ("settle", 2.0),
                "hold",
                "start_key",
                "start",
                "fresh_frames",
                "resume",
            ],
        )
        self.robot.close.assert_called_once()
        self.engine.stop.assert_called_once()

    def test_q_at_home_cancels_before_policy_start(self) -> None:
        with patch("builtins.input", return_value="q"):
            self._run()
        self._assert_cancelled_at_home()

    def test_failed_home_prevents_policy_start_and_closes_devices(self) -> None:
        self.align.side_effect = TimeoutError("home failed")
        with self.assertRaisesRegex(TimeoutError, "home failed"):
            self._run()
        self.engine.start.assert_not_called()
        self.engine.resume.assert_not_called()
        self.robot.wait_for_fresh_observation.assert_not_called()
        self.robot.close.assert_called_once()
        self.engine.stop.assert_called_once()

    def test_recorder_starts_before_devices_and_drains_after_stop_even_on_cleanup_error(self) -> None:
        order: list[str] = []
        writer = MagicMock()
        writer.close.side_effect = lambda: order.append("drain")
        self.robot.connect.side_effect = lambda: order.append("connect")

        def close_robot() -> None:
            order.append("stop_robot")
            raise RuntimeError("cleanup failed")

        self.robot.close.side_effect = close_robot
        self.engine.stop.side_effect = lambda: order.append("stop_engine")

        def create_writer(*args: object, **kwargs: object) -> MagicMock:
            order.append("start_writer")
            return writer

        with (
            patch("deploy_dualarm.DeploymentLog", side_effect=create_writer),
            patch("builtins.input", return_value=""),
            self.assertRaisesRegex(RuntimeError, "cleanup failed"),
        ):
            self._run()
        self.assertEqual(order, ["start_writer", "connect", "stop_robot", "stop_engine", "drain"])

    def test_eof_at_home_cancels_before_policy_start(self) -> None:
        with patch("builtins.input", side_effect=EOFError):
            self._run()
        self._assert_cancelled_at_home()

    def test_interrupt_at_home_closes_devices_without_policy_start(self) -> None:
        with patch("builtins.input", side_effect=KeyboardInterrupt), self.assertRaises(KeyboardInterrupt):
            self._run()
        self._assert_cancelled_at_home()

    def test_invalid_start_text_keeps_waiting_until_enter(self) -> None:
        answers = iter(["not ready", ""])

        def press_start(prompt: str) -> str:
            self.engine.start.assert_not_called()
            self.engine.resume.assert_not_called()
            self.robot.send_action.assert_not_called()
            return next(answers)

        with patch("builtins.input", side_effect=press_start) as keys, redirect_stdout(io.StringIO()):
            self._run()
        self.assertEqual(keys.call_count, 2)
        self.engine.resume.assert_called_once()

    def test_step_mode_waits_for_enter_after_home_and_dispatches_one_action(self) -> None:
        answers = iter(["", "q"])
        observation = dict.fromkeys(ACTION_NAMES, 0.0)

        def press_step(prompt: str) -> str:
            self.assertIn("单步", prompt)
            self.assertIn("align", self.order)
            if keys.call_count == 1:
                dispatch.assert_not_called()
                self.engine.resume.assert_not_called()
                self.robot.send_action.assert_not_called()
            return next(answers)

        with (
            patch(
                "deploy_dualarm.dispatch_single_step", return_value=(observation, np.zeros(16), 0.0)
            ) as dispatch,
            patch("builtins.input", side_effect=press_step) as keys,
        ):
            self._run("step")
        self.assertEqual(keys.call_count, 2)
        dispatch.assert_called_once()
        self.robot.wait_for_target.assert_called_once()
        self.robot.close.assert_called_once()
        self.engine.stop.assert_called_once()


class TestMockedHardware(unittest.TestCase):
    def setUp(self):
        self.deploy_cfg = mock_configuration()
        self.cfg = self.deploy_cfg["hardware"]
        self.created = []
        self.created_grippers = []
        created = self.created
        created_grippers = self.created_grippers

        class Robot:
            def __init__(self, serial, whitelist):
                self.serial, self.commands, self.stops = serial, [], 0
                self.events = []
                self.current_mode, self.active = "IDLE", False
                self.status = "NOT_ENABLED"
                self.q = [0.1 if serial.endswith("063906") else -0.1] * 7
                created.append(self)

            def fault(self):
                return False

            def connected(self):
                return True

            def operational(self):
                return self.active

            def operational_status(self):
                return SimpleNamespace(name=self.status)

            def Enable(self):
                self.active = True
                self.status = "READY"

            def info(self):
                return SimpleNamespace(DoF=7, q_min=[-3.0] * 7, q_max=[3.0] * 7, tau_max=[10.0] * 7)

            def SwitchMode(self, mode):
                self.current_mode = mode
                self.events.append(("mode", mode))

            def SetMaxContactTorque(self, limits):
                self.events.append(("contact_torque", self.current_mode, list(limits)))

            def mode(self):
                return self.current_mode

            def states(self):
                return SimpleNamespace(q=self.q)

            def SendJointPosition(self, q, dq, vel, acc):
                self.commands.append((q, dq, vel, acc))
                self.events.append(("position", list(q)))
                self.q = list(q)

            def Stop(self):
                self.stops += 1

        class Tool:
            def __init__(self, robot):
                pass

            def list(self) -> list[str]:
                return ["Flange", "Teleop-GN01"]

            def name(self) -> str:
                return "Teleop-GN01"

            def Switch(self, name):
                pass

        class Device:
            def __init__(self, robot):
                pass

            def connected(self, name):
                return True

            def enabled(self, name):
                return True

        class Gripper:
            def __init__(self, robot):
                self.commands = []
                self.robot = robot
                self.joint_positions_at_commands = []
                self.width = 0.05
                self.init_calls = 0
                created_grippers.append(self)

            def Init(self):
                self.init_calls += 1

            def Enable(self, name):
                pass

            def params(self):
                return SimpleNamespace(min_width=0.0, max_width=0.1)

            def states(self):
                return SimpleNamespace(width=self.width, is_moving=False, force=0.0)

            def Move(self, width, speed, force):
                self.commands.append((width, speed, force))
                self.joint_positions_at_commands.append(list(self.robot.q))
                self.width = width

        class Camera:
            def __init__(self, rs, serial, cfg):
                pass

            def latest(self):
                return np.zeros((224, 224, 3), dtype=np.uint8), time.monotonic()

            def close(self):
                pass

        sdk = SimpleNamespace(
            Robot=Robot,
            Tool=Tool,
            Device=Device,
            Gripper=Gripper,
            Mode=SimpleNamespace(
                NRT_JOINT_POSITION="NRT_JOINT_POSITION",
                NRT_JOINT_IMPEDANCE="NRT_JOINT_IMPEDANCE",
            ),
        )
        self.addCleanup(patch.stopall)
        patch.dict(sys.modules, {"flexivrdk": sdk, "pyrealsense2": SimpleNamespace()}).start()
        patch("flexiv_deploy_hardware.LatestCamera", Camera).start()
        self.robot = FlexivDualArm(
            self.cfg,
            start_pose=read_json(Path(__file__).resolve().parents[1] / "configs/deploy_config.json")[
                "start_pose"
            ],
        )
        self.addCleanup(self.robot.close)

    def test_only_followers_absolute_commands_gripper_force_and_hold(self):
        self.robot.connect()
        self.assertEqual([robot.serial for robot in self.created], ["Rizon4-063906", "Rizon4R-062106"])
        self.assertFalse(any(gripper.commands for gripper in self.robot.grippers.values()))
        initial_commands = len(self.created[0].commands)
        self.robot.hold()
        self.assertEqual(len(self.created[0].commands), initial_commands)
        obs = self.robot.get_observation()
        target = np.array([obs[name] for name in ACTION_NAMES])
        target[0], target[7], target[8], target[15] = 0.12, 0.08, -0.08, 0.03
        self.robot.send_action(target)
        self.assertAlmostEqual(self.created[0].commands[-1][0][0], 0.12)
        self.assertAlmostEqual(self.created[1].commands[-1][0][0], -0.08)
        np.testing.assert_allclose(self.robot.grippers["left"].commands[-1], (0.08, 0.2, -65.0))
        np.testing.assert_allclose(self.robot.grippers["right"].commands[-1], (0.03, 0.2, 45.0))
        before_hold = len(self.created[0].commands)
        self.robot.hold()
        self.robot.hold()
        self.assertEqual(len(self.created[0].commands), before_hold + 1)
        self.assertEqual(len(self.created[1].commands), before_hold + 1)
        self.assertEqual(len(self.robot.grippers["left"].commands), 1)
        self.assertAlmostEqual(self.created[0].commands[-1][0][0], 0.12)

    def test_gripper_init_waits_for_delayed_motion_and_updates_widths(self) -> None:
        self.robot.connect(with_cameras=False)
        clock = [0.0]

        def sleep(seconds: float) -> None:
            clock[0] += seconds

        def states():
            return SimpleNamespace(width=0.02, is_moving=8.0 <= clock[0] < 12.0, force=0.0)

        counts = [len(arm.commands) for arm in self.created]
        with (
            patch("flexiv_deploy_hardware.time.monotonic", side_effect=lambda: clock[0]),
            patch("flexiv_deploy_hardware.time.sleep", side_effect=sleep),
            patch.object(sys.modules["flexivrdk"].Gripper, "states", side_effect=states),
        ):
            self.robot.initialize_grippers()
        self.assertGreaterEqual(clock[0], 12.5)
        self.assertEqual([g.init_calls for g in self.created_grippers], [1, 1])
        self.assertEqual([len(arm.commands) for arm in self.created], counts)
        for width in self.robot._last_gripper.values():
            self.assertAlmostEqual(width, 0.02)
        self.assertTrue(all(not g.commands for g in self.created_grippers))

    def test_gripper_init_motion_timeout(self) -> None:
        self.robot.connect(with_cameras=False)
        clock = [0.0]

        def sleep(seconds: float) -> None:
            clock[0] += seconds

        with (
            patch("flexiv_deploy_hardware.time.monotonic", side_effect=lambda: clock[0]),
            patch("flexiv_deploy_hardware.time.sleep", side_effect=sleep),
            patch.object(
                sys.modules["flexivrdk"].Gripper,
                "states",
                return_value=SimpleNamespace(width=0.05, is_moving=True),
            ),
            self.assertRaisesRegex(TimeoutError, "Init settling timed out"),
        ):
            self.robot.initialize_grippers()
        self.assertEqual([g.init_calls for g in self.created_grippers], [1, 1])
        self.assertTrue(all(not g.commands for g in self.created_grippers))

    def test_active_tool_does_not_trigger_redundant_switch(self) -> None:
        def read_before_enable() -> str:
            self.assertFalse(self.created[-1].active)
            return "Teleop-GN01"

        with (
            patch.object(sys.modules["flexivrdk"].Tool, "name", side_effect=read_before_enable) as name,
            patch.object(sys.modules["flexivrdk"].Tool, "Switch") as switch,
        ):
            self.robot.connect(with_cameras=False)
        switch.assert_not_called()
        self.assertEqual(name.call_count, 2)
        self.assertEqual(len(self.created), 2)

    def test_transient_tool_list_and_name_no_reply_are_retried_before_enable(self) -> None:
        for method, response in (("list", ["Flange", "Teleop-GN01"]), ("name", "Teleop-GN01")):
            with self.subTest(method=method):
                self.created.clear()
                with (
                    patch.object(
                        sys.modules["flexivrdk"].Tool,
                        method,
                        side_effect=[RuntimeError("No reply from the robot"), response, response],
                    ) as query,
                    patch("flexiv_deploy_hardware.time.sleep") as sleep,
                ):
                    self.robot.connect(with_cameras=False)
                self.assertEqual(query.call_count, 3)
                sleep.assert_called_once_with(0.25)
                self.assertEqual(len(self.created), 2)
                self.robot.close()

    def test_permanent_tool_no_reply_times_out_before_enable_and_stops_follower(self) -> None:
        self.cfg["connect_timeout_s"] = 0.5
        clock = [0.0]

        def sleep(seconds: float) -> None:
            clock[0] += seconds

        with (
            patch.object(
                sys.modules["flexivrdk"].Tool,
                "list",
                side_effect=RuntimeError("[flexiv::rdk::Tool::list] No reply from the robot"),
            ) as query,
            patch("flexiv_deploy_hardware.time.monotonic", side_effect=lambda: clock[0]),
            patch("flexiv_deploy_hardware.time.sleep", side_effect=sleep),
            self.assertRaisesRegex(TimeoutError, r"left.*Rizon4-063906.*Tool.list.*0.5 s"),
        ):
            self.robot.connect(with_cameras=False)
        self.assertEqual(query.call_count, 3)
        self.assertFalse(self.created[0].active)
        self.assertFalse(self.created[0].commands)
        self.assertEqual(self.created[0].stops, 1)
        self.assertFalse(self.created_grippers)

    def test_other_tool_query_errors_are_not_retried(self) -> None:
        with (
            patch.object(
                sys.modules["flexivrdk"].Tool, "list", side_effect=RuntimeError("Invalid SDK state")
            ) as query,
            patch("flexiv_deploy_hardware.time.sleep") as sleep,
            self.assertRaisesRegex(RuntimeError, "Invalid SDK state"),
        ):
            self.robot.connect(with_cameras=False)
        query.assert_called_once()
        sleep.assert_not_called()
        self.assertFalse(self.created[0].active)
        self.assertFalse(self.created[0].commands)

    def test_fault_or_disconnect_during_tool_retry_aborts_without_enable(self) -> None:
        for method, value in (("fault", True), ("connected", False)):
            with self.subTest(method=method):
                self.created.clear()

                def lose_robot(seconds: float, state_method: str = method, reported: bool = value) -> None:
                    setattr(self.created[0], state_method, lambda: reported)

                with (
                    patch.object(
                        sys.modules["flexivrdk"].Tool,
                        "list",
                        side_effect=RuntimeError("No reply from the robot"),
                    ) as query,
                    patch("flexiv_deploy_hardware.time.sleep", side_effect=lose_robot),
                    self.assertRaisesRegex(RuntimeError, "disconnected or faulted"),
                ):
                    self.robot.connect(with_cameras=False)
                query.assert_called_once()
                self.assertFalse(self.created[0].active)
                self.assertFalse(self.created[0].commands)
                self.assertEqual(self.created[0].stops, 1)

    def test_tool_switch_write_is_not_retried_on_no_reply(self) -> None:
        with (
            patch.object(sys.modules["flexivrdk"].Tool, "name", return_value="Flange"),
            patch.object(
                sys.modules["flexivrdk"].Tool, "Switch", side_effect=RuntimeError("No reply from the robot")
            ) as switch,
            patch("flexiv_deploy_hardware.time.sleep") as sleep,
            self.assertRaisesRegex(RuntimeError, "No reply"),
        ):
            self.robot.connect(with_cameras=False)
        switch.assert_called_once_with("Teleop-GN01")
        sleep.assert_not_called()
        self.assertFalse(self.created[0].commands)
        self.assertEqual(self.created[0].stops, 1)

    def test_missing_tool_reports_available_names_before_enabling(self) -> None:
        self.cfg["arms"]["left"]["tool"] = "MissingTool"
        with (
            patch.object(sys.modules["flexivrdk"].Tool, "Switch") as switch,
            self.assertRaisesRegex(ValueError, r"left.*MissingTool.*available tools=.*Teleop-GN01"),
        ):
            self.robot.connect(with_cameras=False)
        switch.assert_not_called()
        self.assertEqual(len(self.created), 1)
        self.assertFalse(self.created[0].active)
        self.assertFalse(self.created[0].commands)
        self.assertFalse(self.created_grippers)
        self.assertEqual(self.created[0].stops, 1)

    def test_different_active_tool_switches_to_configured_tool(self) -> None:
        with (
            patch.object(sys.modules["flexivrdk"].Tool, "name", return_value="Flange"),
            patch.object(sys.modules["flexivrdk"].Tool, "Switch") as switch,
        ):
            self.robot.connect(with_cameras=False)
        self.assertEqual(switch.call_args_list, [call("Teleop-GN01"), call("Teleop-GN01")])

    def test_arm_only_connection_does_not_open_cameras(self):
        self.robot.connect(with_cameras=False)
        self.assertFalse(self.robot.cameras)
        self.assertEqual([robot.serial for robot in self.created], ["Rizon4-063906", "Rizon4R-062106"])

    def test_null_or_missing_contact_limits_do_not_override_robot_settings(self):
        del self.cfg["arms"]["right"]["max_contact_torque_nm"]
        self.robot.connect(with_cameras=False)
        for robot in self.created:
            self.assertEqual([event[0] for event in robot.events], ["mode", "position"])

    def test_per_arm_contact_limits_applied_in_impedance_mode_before_first_position(self):
        self.cfg["control_mode"] = "NRT_JOINT_IMPEDANCE"
        left, right = list(range(7)), [10.0] * 7
        self.cfg["arms"]["left"]["max_contact_torque_nm"] = left
        self.cfg["arms"]["right"]["max_contact_torque_nm"] = right
        self.robot.connect(with_cameras=False)
        for robot, limits in zip(self.created, (left, right), strict=True):
            self.assertEqual([event[0] for event in robot.events], ["mode", "contact_torque", "position"])
            self.assertEqual(robot.events[1], ("contact_torque", "NRT_JOINT_IMPEDANCE", limits))

    def test_incompatible_contact_limits_fail_before_any_device_is_claimed(self):
        self.cfg["arms"]["right"]["max_contact_torque_nm"] = [1.0] * 7
        with (
            patch("flexiv_deploy_hardware.LatestCamera") as camera,
            self.assertRaisesRegex(ValueError, "right.*NRT_JOINT_IMPEDANCE"),
        ):
            self.robot.connect()
        camera.assert_not_called()
        self.assertFalse(self.created)

    def test_contact_limit_above_robot_rating_fails_before_enable_or_position_command(self):
        self.cfg["control_mode"] = "NRT_JOINT_IMPEDANCE"
        self.cfg["arms"]["left"]["max_contact_torque_nm"] = [1.0] * 6 + [10.1]
        with self.assertRaisesRegex(ValueError, r"left.*tau_max: A7=10.1 > 10 Nm"):
            self.robot.connect(with_cameras=False)
        self.assertEqual(len(self.created), 1)
        self.assertFalse(self.created[0].active)
        self.assertFalse(self.created[0].commands)
        self.assertFalse(self.created_grippers)

    def test_failed_torque_setting_stops_connection_before_position_command(self):
        self.cfg["control_mode"] = "NRT_JOINT_IMPEDANCE"
        self.cfg["arms"]["left"]["max_contact_torque_nm"] = [1.0] * 7
        with (
            patch.object(
                sys.modules["flexivrdk"].Robot, "SetMaxContactTorque", side_effect=RuntimeError("SDK refused")
            ),
            self.assertRaisesRegex(RuntimeError, "SDK refused"),
        ):
            self.robot.connect(with_cameras=False)
        self.assertFalse(self.created[0].commands)
        self.assertEqual(self.created[0].stops, 1)

    def test_stale_camera_error_identifies_camera_serial_and_frame_age(self):
        self.robot.connect()
        with (
            patch("flexiv_deploy_hardware.time.monotonic", return_value=12.25),
            patch.object(
                self.robot.cameras["image2"],
                "latest",
                return_value=(np.zeros((224, 224, 3), dtype=np.uint8), 12.0),
            ),
            self.assertRaisesRegex(RuntimeError, r"Stale.*limit=200.0 ms.*image2 \(840412062188\): 250.0 ms"),
        ):
            self.robot.get_observation()

    def test_fresh_camera_ages_are_available_in_telemetry(self):
        self.robot.connect()
        self.robot.get_observation()
        self.assertEqual(set(self.robot.camera_ages_s), {"image", "image2", "image3"})
        self.assertTrue(all(0 <= age < 0.2 for age in self.robot.camera_ages_s.values()))

    def test_startup_waits_for_new_frames_without_sending_actions(self):
        self.robot.connect()
        commands_before = [len(robot.commands) for robot in self.created]
        observation = {"ready": True}
        self.robot.observation_time = 10.1
        with (
            patch("flexiv_deploy_hardware.time.monotonic", return_value=10.0),
            patch("flexiv_deploy_hardware.time.sleep") as sleep,
            patch.object(
                self.robot,
                "get_observation",
                side_effect=[
                    CameraFreshnessError("image2: 1682.8 ms"),
                    observation,
                ],
            ),
        ):
            self.assertIs(self.robot.wait_for_fresh_observation(1.0), observation)
        sleep.assert_called_once_with(0.01)
        self.assertEqual([len(robot.commands) for robot in self.created], commands_before)

    def test_startup_timeout_retains_stale_camera_details(self):
        self.robot.connect()
        with (
            patch("flexiv_deploy_hardware.time.monotonic", side_effect=[10.0, 11.1]),
            patch.object(
                self.robot, "get_observation", side_effect=CameraFreshnessError("image2: 1682.8 ms")
            ),
            self.assertRaisesRegex(TimeoutError, "image2: 1682.8 ms"),
        ):
            self.robot.wait_for_fresh_observation(1.0)

    def test_startup_does_not_retry_reader_or_robot_failure(self):
        self.robot.connect()
        with (
            patch.object(self.robot, "get_observation", side_effect=RuntimeError("RealSense reader failed")),
            self.assertRaisesRegex(RuntimeError, "RealSense reader failed"),
        ):
            self.robot.wait_for_fresh_observation(1.0)

    def test_manual_left_fails_before_enable_and_skips_unavailable_stop(self):
        sdk = sys.modules["flexivrdk"]
        robot_class = sdk.Robot

        def manual_robot(serial, whitelist):
            robot = robot_class(serial, whitelist)
            robot.status = "IN_MANUAL_MODE"
            return robot

        with (
            patch.object(sdk, "Robot", side_effect=manual_robot),
            self.assertRaisesRegex(RuntimeError, "left follower Rizon4-063906.*Manual mode"),
        ):
            self.robot.connect(with_cameras=False)
        self.assertEqual(len(self.created), 1)
        self.assertFalse(self.created[0].active)
        self.assertEqual(self.created[0].stops, 0)
        self.assertFalse(self.created[0].commands)
        self.assertFalse(self.created_grippers)

    def test_gripper_close_wait_reports_the_side_that_remains_open(self):
        self.robot.connect(with_cameras=False)
        self.robot.grippers["left"].width = 0.004
        self.robot.grippers["right"].width = 0.0
        with self.assertRaisesRegex(TimeoutError, "left: width=0.0040 m"):
            self.robot.wait_for_grippers_closed(tolerance_m=0.001, timeout_s=0.01)

    def test_opening_timeout_distinguishes_zero_width_feedback_from_joint_error(self):
        self.robot.connect(with_cameras=False)
        target = self.robot.read_state()
        target[[7, 15]] = 0.1
        for gripper in self.created_grippers:
            gripper.width = 0.0
        with (
            patch("flexiv_deploy_hardware.time.monotonic", side_effect=[10.0, 11.0]),
            self.assertRaises(TimeoutError) as raised,
        ):
            self.robot.wait_for_target(target, 0.5, operation="Opening both grippers")
        message = str(raised.exception)
        self.assertIn("Opening both grippers", message)
        self.assertIn("pending=grippers;", message)
        for side in ("left", "right"):
            self.assertIn(f"{side}: joint_error=0.0000 rad", message)
        self.assertEqual(message.count("gripper target=100.00 mm, measured=0.00 mm"), 2)

    def test_target_timeout_reports_joint_failure_with_correct_gripper_feedback(self):
        self.robot.connect(with_cameras=False)
        target = self.robot.read_state()
        target[8] += 0.1
        with (
            patch("flexiv_deploy_hardware.time.monotonic", side_effect=[10.0, 11.0]),
            self.assertRaisesRegex(TimeoutError, r"pending=joints;.*right: joint_error=0.1000 rad"),
        ):
            self.robot.wait_for_target(target, 0.5)

    def test_delayed_width_feedback_is_waited_for_without_resending_motion(self):
        self.robot.connect(with_cameras=False)
        target = self.robot.read_state()
        target[[7, 15]] = 0.1
        before = [len(robot.commands) for robot in self.created]
        with (
            patch.object(self.robot, "read_state", side_effect=[self.robot.read_state(), target]),
            patch("flexiv_deploy_hardware.time.sleep"),
        ):
            result = self.robot.wait_for_target(target, 1.0)
        np.testing.assert_allclose(result, target)
        self.assertEqual([len(robot.commands) for robot in self.created], before)

    def test_go_home_sends_close_even_when_both_widths_are_inside_deadband(self):
        self.robot.connect(with_cameras=False)
        for side, width in (("left", 0.00072), ("right", 0.0)):
            self.robot.grippers[side].width = width
            self.robot._last_gripper[side] = width
        target = self.robot.read_state()
        target[[7, 15]] = 0.0
        align_start_pose(
            self.robot, (target, {"scene": "scene_0001", "timestamp_ms": 150.0}), close_grippers=True
        )
        for gripper in self.created_grippers:
            self.assertEqual(gripper.commands, [(0.0, self.cfg["gripper_speed_m_s"], 45.0)])

    def test_go_home_closes_grippers_and_moves_fake_followers_without_prompt(self):
        target = pack_state([0.22] * 7, 0.05, [-0.22] * 7, 0.05)
        start_pose = (target, {"scene": "scene_0001", "timestamp_ms": 150.0})
        clock = [0.0]

        def sleep(seconds: float) -> None:
            clock[0] += seconds

        with (
            patch("flexiv_deploy_hardware.time.monotonic", side_effect=lambda: clock[0]),
            patch("flexiv_deploy_hardware.time.sleep", side_effect=sleep),
            patch.object(sys, "argv", ["goHome.py"]),
            patch("goHome.read_json", return_value=self.deploy_cfg),
            patch("goHome.read_start_pose", return_value=start_pose),
            patch("goHome.read_home_tcp", return_value=np.zeros((2, 7))),
            patch("go_home_path.plan_home"),
            patch(
                "go_home_path.execute_home",
                side_effect=lambda robot, cfg, target, plan: robot.send_action(target),
            ),
            patch("goHome.require_compatible_rdk"),
            patch("builtins.input", side_effect=AssertionError("unexpected prompt")),
        ):
            go_home_main()
        self.assertEqual([robot.serial for robot in self.created], ["Rizon4-063906", "Rizon4R-062106"])
        np.testing.assert_allclose(self.created[0].q, target[:7], atol=0.02)
        np.testing.assert_allclose(self.created[1].q, target[8:15], atol=0.02)
        self.assertGreater(len(self.created[0].commands), 1)
        self.assertGreater(len(self.created[1].commands), 1)
        self.assertEqual([gripper.width for gripper in self.created_grippers], [0.0, 0.0])
        self.assertEqual([gripper.init_calls for gripper in self.created_grippers], [0, 0])
        for gripper, joint_target in zip(self.created_grippers, (target[:7], target[8:15]), strict=True):
            self.assertEqual(gripper.commands[0], (0.1, 0.2, -65.0))
            self.assertEqual(gripper.commands[-1], (0.0, 0.2, 45.0))
            np.testing.assert_allclose(gripper.joint_positions_at_commands[-1], joint_target, atol=0.02)
        self.assertTrue(all(robot.stops == 1 for robot in self.created))

    def test_bad_right_target_cannot_send_a_left_command(self):
        self.robot.connect()
        count = len(self.created[0].commands)
        target = self.robot.read_state()
        target[8] = 10
        with self.assertRaises(ValueError):
            self.robot.send_action(target)
        self.assertEqual(len(self.created[0].commands), count)

    def test_initial_pose_error_identifies_joint_without_moving(self):
        self.robot.connect()
        target = self.robot.read_state()
        target[0] += 1.2
        target[8] -= 1.1
        before = [len(robot.commands) for robot in self.created]
        with self.assertRaisesRegex(ValueError, "left_joint_1.*right_joint_1") as raised:
            self.robot.check_initial_pose(target)
        self.assertIn("limit 1.000 rad", str(raised.exception))
        self.assertIn("first_frame=1.300", str(raised.exception))
        self.assertEqual([len(robot.commands) for robot in self.created], before)

    def test_initial_pose_distance_limit_can_be_disabled_for_gohome(self):
        self.robot.connect()
        target = self.robot.read_state()
        target[0] += 1.2
        measured, distance = self.robot.check_initial_pose(target, enforce_distance_limit=False)
        np.testing.assert_array_equal(measured, self.robot.read_state())
        self.assertAlmostEqual(distance, 1.2, places=6)

    def test_partial_connect_failure_stops_already_connected_followers(self):
        calls = 0

        def fail_second(predicate, operation):
            nonlocal calls
            calls += 1
            if calls == 4:
                raise RuntimeError("mock second arm failure")
            if not predicate():
                raise AssertionError("mock not ready")

        with (
            patch.object(self.robot, "_wait", side_effect=fail_second),
            self.assertRaisesRegex(RuntimeError, "mock second arm failure"),
        ):
            self.robot.connect()
        self.assertEqual(len(self.created), 2)
        self.assertTrue(all(robot.stops == 1 for robot in self.created))
        self.assertFalse(self.robot.robots)
        self.assertFalse(self.robot.cameras)

    def test_initial_alignment_uses_small_joint_waypoints_then_grippers(self):
        self.robot.connect()
        initial = self.robot.read_state()
        target = initial.copy()
        target[0] += 0.12
        target[8] -= 0.11
        target[7], target[15] = 0.02, 0.08
        reached = self.robot.move_to_initial_pose(target)
        np.testing.assert_allclose(reached, target, atol=0.02)
        for arm in self.created:
            positions = [np.asarray(command[0]) for command in arm.commands]
            self.assertLessEqual(
                max(np.max(np.abs(a - b)) for a, b in zip(positions, positions[1:], strict=False)),
                0.05 + 1e-6,
            )
        self.assertEqual(len(self.robot.grippers["left"].commands), 1)
        self.assertEqual(len(self.robot.grippers["right"].commands), 1)

    def test_initial_alignment_rejects_large_unexpected_move(self):
        self.robot.connect()
        target = self.robot.read_state()
        target[0] += 1.1
        count = len(self.created[0].commands)
        with self.assertRaisesRegex(ValueError, "manually pre-position"):
            self.robot.move_to_initial_pose(target)
        self.assertEqual(len(self.created[0].commands), count)

    def test_initial_alignment_accepts_previous_0888_rad_offset(self):
        self.robot.connect()
        target = self.robot.read_state()
        target[0] += 0.888
        reached = self.robot.move_to_initial_pose(target)
        np.testing.assert_allclose(reached, target, atol=0.02)
        positions = [np.asarray(command[0]) for command in self.created[0].commands]
        self.assertLessEqual(
            max(np.max(np.abs(a - b)) for a, b in zip(positions, positions[1:], strict=False)),
            0.05 + 1e-6,
        )


if __name__ == "__main__":
    unittest.main()
