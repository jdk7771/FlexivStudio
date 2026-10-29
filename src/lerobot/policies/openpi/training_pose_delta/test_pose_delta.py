"""CPU checks for geometry, actual conversion, normalization and OpenPI gradients."""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scipy.spatial.transform import Rotation
import torch

from pose_math import apply_pose_delta, interpolate_tcp, pose_delta, state_features
from prepare_dataset import action_chunks, prepare
from runtime import Collator, load_pose_stats, model_config, pose_loss
from train import read_config, validate_ddp_config


def test_pi05_two_gpu_batch16_config_and_sampler():
    from torch.utils.data import DataLoader, DistributedSampler

    config = json.loads((Path(__file__).parent / "configs/dualarm_fold_towel_pose_delta_no_rtc_full.json").read_text())
    assert model_config(config).pi05 is True
    assert Path(config["policy"]["pretrained_path"]).name == "pi05_base"
    ranks = validate_ddp_config(config, 2)
    assert config["batch_size"] == 16
    assert config["policy"]["rtc_training_max_delay"] == 0
    assert config["policy"]["rtc_config"] is None
    batches = []
    for rank in range(ranks):
        sampler = DistributedSampler(range(64), num_replicas=ranks, rank=rank, seed=1000)
        loader = DataLoader(range(64), sampler=sampler, batch_size=config["batch_size"], drop_last=True)
        batches.append(list(loader))
    assert all(batch.shape == (16,) for rank_batches in batches for batch in rank_batches)
    assert set(torch.cat(batches[0]).tolist()).isdisjoint(torch.cat(batches[1]).tolist())
    assert set(torch.cat(batches[0] + batches[1]).tolist()) == set(range(64))
    with pytest.raises(ValueError, match="WORLD_SIZE"):
        validate_ddp_config(config, 4)


@pytest.mark.parametrize("batch_size", [0, -1, 1.5, True])
def test_invalid_per_rank_batch_rejected(batch_size):
    with pytest.raises(ValueError, match="batch_size"):
        validate_ddp_config({"parallelism": {"dp_replicate": 2, "dp_shard": 1}, "batch_size": batch_size})


def test_launcher_uses_two_ranks_and_rejects_mismatch():
    script = Path(__file__).parent / "train_dualarm_fold_towel_pose_delta_no_rtc_full.sh"
    env = dict(os.environ, OPENPI_PYTHON=sys.executable, CUDA_VISIBLE_DEVICES="0,1", MASTER_PORT="29526")
    env.pop("OPENPI_CONFIG", None)
    result = subprocess.run(["bash", str(script), "--print-command"], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--nproc-per-node=2" in result.stdout and "--master-port=29526" in result.stdout
    env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    result = subprocess.run(["bash", str(script), "--print-command"], env=env, capture_output=True, text=True)
    assert result.returncode != 0 and "must match config" in result.stderr


def pose(xyz=(0, 0, 0), rot=None, gripper=0.03):
    if rot is None:
        rot = Rotation.identity()
    arm = np.r_[xyz, rot.as_quat()[[3, 0, 1, 2]], gripper]
    return np.tile(arm, 2).astype(np.float32)


@pytest.mark.parametrize("frame", ["base", "tcp"])
def test_noncommuting_rotation_and_translation_roundtrip(frame):
    origin = Rotation.from_euler("z", 90, degrees=True)
    relative = Rotation.from_euler("x", 30, degrees=True)
    target = relative * origin if frame == "base" else origin * relative
    current = pose((2, 3, 4), origin)
    future = pose((3, 3, 4), target, 0.08)
    delta = pose_delta(current, future, frame)
    np.testing.assert_allclose(delta[:3], [1, 0, 0] if frame == "base" else [0, -1, 0], atol=1e-6)
    np.testing.assert_allclose(delta[3:6], [np.pi / 6, 0, 0], atol=1e-6)
    assert delta[6] == pytest.approx(0.08)  # absolute target, not opening difference
    restored = apply_pose_delta(current, delta, frame)
    np.testing.assert_allclose(state_features(restored), state_features(future), atol=1e-6)


def test_chunk_has_one_current_origin_and_independent_arms():
    current = pose((1, 0, 0))
    targets = np.stack([pose((x, 0, 0)) for x in (2, 4, 7)])
    targets[:, 8] += 10
    delta = pose_delta(current, targets)
    np.testing.assert_allclose(delta[:, 0], [1, 3, 6])
    np.testing.assert_allclose(delta[:, 7], [11, 13, 16])
    batches = pose_delta(np.stack([current, current])[:, None], np.stack([targets, targets]))
    np.testing.assert_allclose(batches, np.stack([delta, delta]))


def test_quaternion_sign_and_pi_wrap():
    current = pose(rot=Rotation.from_euler("z", 179, degrees=True))
    target = pose(rot=Rotation.from_euler("z", -179, degrees=True))
    changed_sign = target.copy()
    changed_sign[3:7] *= -1
    changed_sign[11:15] *= -1
    np.testing.assert_allclose(state_features(target), state_features(changed_sign), atol=1e-6)
    np.testing.assert_allclose(pose_delta(current, target), pose_delta(current, changed_sign), atol=1e-6)
    assert pose_delta(current, target)[5] == pytest.approx(np.deg2rad(2), abs=1e-6)


def test_slerp_shortest_path_and_no_extrapolation():
    samples = np.stack([pose(rot=Rotation.from_euler("z", a, degrees=True))[:7] for a in (170, -170)])
    midpoint = interpolate_tcp(np.array([1000, 2000]), samples, np.array([1500]))[0]
    rotation = Rotation.from_quat(midpoint[[4, 5, 6, 3]])
    np.testing.assert_allclose(rotation.apply([1, 0, 0]), [-1, 0, 0], atol=1e-6)
    with pytest.raises(ValueError, match="extrapolate"):
        interpolate_tcp(np.array([1000, 2000]), samples, np.array([999]))
    samples[0, 3:] = 0
    with pytest.raises(ValueError, match="quaternion"):
        interpolate_tcp(np.array([1000, 2000]), samples, np.array([1500]))


def test_repeat_last_padding_does_not_cross_episode():
    actions = np.stack([pose((x, 0, 0)) for x in (1, 2, 3)])
    chunks = action_chunks(actions, 4)
    np.testing.assert_allclose(chunks[:, :, 0], [[1, 2, 3, 3], [2, 3, 3, 3], [3, 3, 3, 3]])


@pytest.fixture
def converted_dataset(tmp_path):
    """Two small episodes; held-out positions are far outside the training range."""
    config = json.loads((Path(__file__).parent / "configs/dualarm_fold_towel_pose_delta_no_rtc_full.json").read_text())
    aligned, source, output = (tmp_path / name for name in ("joint", "raw", "pose"))
    config["dataset"].update(root=str(output), eval_split=0.5)
    config["policy"]["chunk_size"] = 3
    config["preparation"].update(source_root=str(source), aligned_joint_root=str(aligned), max_lowdim_gap_ms=101)
    (aligned / "meta/episodes/chunk-000").mkdir(parents=True)
    (aligned / "data/chunk-000").mkdir(parents=True)
    (aligned / "videos").mkdir()
    (source / "scene_0001/lowdim").mkdir(parents=True)
    times = np.arange(31) * 100
    tcp = np.tile(pose()[:7], (31, 1))
    tcp[:, 0] = np.where(times < 1000, times / 1000, 100 + times / 1000)
    with h5py.File(source / "scene_0001/lowdim/lowdim.h5", "w") as h5:
        h5["timestamp"] = times
        for serial in ("left", "right"):
            h5[f"tcp_pose_{serial}"] = tcp
            h5[f"ee_state_{serial}"] = np.full((31, 1), 0.03)
            h5[f"ee_command_{serial}"] = np.full((31, 1), 0.08)
    info = {"features": {key: {"dtype": "float32", "shape": [16], "names": ["joint"] * 16}
                         for key in ("action", "observation.state")}, "total_frames": 6,
            "total_episodes": 2, "fps": 10}
    (aligned / "meta/info.json").write_text(json.dumps(info))
    (aligned / "meta/stats.json").write_text("{}")
    segments, rows = [], []
    for index, start in enumerate((0, 2000)):
        segments.append(dict(episode_index=index, source_start_ms=start, source_end_ms=start + 200, frames=3))
        rows.append(dict(episode_index=index, tasks=["fold"], length=3,
                         dataset_from_index=index * 3, dataset_to_index=(index + 1) * 3))
    pq.write_table(pa.Table.from_pylist(rows), aligned / "meta/episodes/chunk-000/file-000.parquet")
    manifest = {"status": "complete", "arms": {"left": "left", "right": "right"},
                "scenes": [{"scene": "scene_0001", "segments": segments}]}
    (aligned / "meta/conversion_manifest.json").write_text(json.dumps(manifest))
    states = np.stack([pose(gripper=0.03)] * 6)
    actions = np.stack([pose(gripper=0.08)] * 6)
    table = pa.table({"observation.state": pa.array(states.tolist(), type=pa.list_(pa.float32(), 16)),
                      "action": pa.array(actions.tolist(), type=pa.list_(pa.float32(), 16)),
                      "index": np.arange(6), "frame_index": [0, 1, 2, 0, 1, 2], "episode_index": [0, 0, 0, 1, 1, 1]})
    pq.write_table(table, aligned / "data/chunk-000/file-000.parquet")
    original = {str(p): p.read_bytes() for p in aligned.rglob("*") if p.is_file()}
    prepare(config)
    assert all(Path(p).read_bytes() == data for p, data in original.items())
    return config, output


def test_conversion_labels_and_train_only_stats(converted_dataset):
    config, output = converted_dataset
    stats = load_pose_stats(config)
    assert stats["train_episodes"] == [0] and stats["eval_episodes"] == [1]
    assert stats["observation.state"]["max"][0] < 1  # held-out episode is at x > 100
    assert stats["action"]["count"] == [9]
    table = pq.read_table(output / "data/chunk-000/file-000.parquet")
    target = np.asarray(table["action"].to_pylist())
    np.testing.assert_allclose(target[:3, 0], [0.1, 0.2, 0.3], atol=1e-6)
    assert (output / "videos").is_symlink()
    changed = copy.deepcopy(config)
    changed["pose_delta"]["delta_frame"] = "tcp"
    with pytest.raises(ValueError, match="transform/horizon"):
        load_pose_stats(changed)
    with pytest.raises(FileExistsError):
        prepare(config)


@pytest.mark.parametrize("batch_size", [1, 16])
def test_collator_14d_actions_20d_state_padding_and_constant_dimensions(converted_dataset, batch_size):
    config, output = converted_dataset
    info = json.loads((output / "meta/info.json").read_text())
    cameras = ["observation.images.image", "observation.images.image2", "observation.images.image3"]
    meta = SimpleNamespace(features=info["features"], camera_keys=cameras[::-1], total_episodes=2,
                           episodes=[{"tasks": ["fold"]}, {"tasks": ["fold"]}])
    collator = Collator(config, meta)
    class Tokenizer:
        def __call__(self, prompts, **kwargs):
            return SimpleNamespace(input_ids=torch.ones(len(prompts), 200, dtype=torch.long),
                                   attention_mask=torch.ones(len(prompts), 200, dtype=torch.long))
    collator.tokenizer = Tokenizer()
    table = pq.read_table(output / "data/chunk-000/file-000.parquet")
    state = torch.tensor(table["observation.state"][0].as_py())
    targets = torch.tensor(table["action"].to_pylist()[:3])
    sample = {"observation.state": state, "action": targets, "task": "fold",
              **{camera: torch.full((3, 4, 4), index * 20, dtype=torch.uint8) for index, camera in enumerate(cameras)}}
    observation, actions = collator([sample] * batch_size)
    assert observation["state"].shape == (batch_size, 32) and actions.shape == (batch_size, 3, 32)
    assert torch.count_nonzero(actions[..., 14:]) == 0
    assert torch.count_nonzero(observation["state"][..., 20:]) == 0
    assert torch.isfinite(actions).all()
    np.testing.assert_allclose(collator.denormalize_actions(actions.numpy()),
                               np.repeat(pose_delta(state.numpy(), targets.numpy())[None], batch_size, axis=0), atol=1e-6)
    assert torch.all(observation["images"]["base_0_rgb"] == -1)


def test_pose_loss_trains_all_timesteps_but_excludes_padding():
    losses = torch.arange(2 * 5 * 32, dtype=torch.float32).reshape(2, 5, 32).requires_grad_()
    value = pose_loss(losses, 14)
    expected = sum(losses[b, t, d].item() for b in range(2) for t in range(5) for d in range(14)) / 140
    assert value.item() == pytest.approx(expected)
    value.backward()
    assert torch.count_nonzero(losses.grad[..., 14:]) == 0
    torch.testing.assert_close(losses.grad[..., :14], torch.full((2, 5, 14), 1 / 140))


@pytest.mark.parametrize("rtc_override", [{"rtc_training_max_delay": 8}, {"rtc_config": {}}])
def test_pose_entry_rejects_rtc_config(tmp_path, rtc_override):
    config = json.loads((Path(__file__).parent / "configs/dualarm_fold_towel_pose_delta_no_rtc_full.json").read_text())
    config["policy"].update(rtc_override)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="disables RTC"):
        read_config(path)


def test_official_tiny_model_pose_backward(monkeypatch):
    import openpi.models_pytorch.gemma_pytorch as gemma
    import openpi.models_pytorch.pi0_pytorch as pi0

    original_vlm, original_expert = gemma.PaliGemmaForConditionalGeneration, gemma.GemmaForCausalLM
    def small_vlm(config):
        config.text_config.vocab_size = config._vocab_size = 512
        config.image_token_index = 511
        config.vision_config.hidden_size = config.vision_config.projection_dim = 32
        config.vision_config.intermediate_size = 64
        config.vision_config.num_attention_heads = 4
        config.vision_config.num_hidden_layers = 1
        return original_vlm(config)
    def small_expert(config):
        config.vocab_size = 512
        return original_expert(config)
    monkeypatch.setattr(gemma, "PaliGemmaForConditionalGeneration", small_vlm)
    monkeypatch.setattr(gemma, "GemmaForCausalLM", small_expert)
    monkeypatch.setattr(pi0._gemma, "get_config", lambda name: SimpleNamespace(
        width=32, mlp_dim=64, num_heads=8, head_dim=4, depth=1, num_kv_heads=1))
    model = pi0.PI0Pytorch(SimpleNamespace(
        pi05=True, paligemma_variant="gemma_2b", action_expert_variant="gemma_300m",
        dtype="float32", action_dim=32, action_horizon=10, pytorch_compile_mode=None,
        image_augmentation=False))
    model.gradient_checkpointing_enable()
    keys = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    observation = SimpleNamespace(images={k: torch.zeros(1, 3, 224, 224) for k in keys},
        image_masks={k: torch.ones(1, dtype=torch.bool) for k in keys}, state=torch.zeros(1, 32),
        tokenized_prompt=torch.ones(1, 4, dtype=torch.long), tokenized_prompt_mask=torch.ones(1, 4, dtype=torch.bool),
        token_ar_mask=None, token_loss_mask=None)
    targets = np.stack([pose((0.01 * k, 0, 0)) for k in range(1, 11)])
    actions = torch.nn.functional.pad(torch.from_numpy(pose_delta(pose(), targets))[None], (0, 18))
    observed_times = []
    original_embed_suffix = model.embed_suffix
    def record_time(state, noisy_actions, time):
        observed_times.append(time.detach().clone())
        return original_embed_suffix(state, noisy_actions, time)
    monkeypatch.setattr(model, "embed_suffix", record_time)
    loss = pose_loss(model(observation, actions), 14)
    assert len(observed_times) == 1 and observed_times[0].shape == (1,)
    assert torch.all(observed_times[0] > 0)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(param.requires_grad for param in model.parameters())
    for parameter in (model.action_in_proj.weight,
                      model.paligemma_with_expert.paligemma.vision_tower.vision_model.embeddings.patch_embedding.weight,
                      model.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
