"""Independent four-GPU full fine-tuning of OpenPI pi0.5 on TCP pose deltas + RTC.

The model is openpi.models_pytorch.pi0_pytorch.PI0Pytorch, never LeRobot's policy.
LeRobot supplies only the existing v3 dataset reader. --check-data is CPU-only.
"""

import argparse
import csv
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import shutil
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
# Use the isolated LeRobot 0.4.4 v3 reader; the surrounding checkout needs transformers 5.

import numpy as np
from safetensors import safe_open
from safetensors.torch import load_model, save_model
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, Subset

from runtime import Collator, learning_rate, load_pose_stats, model_config, observation_to_device, rtc_loss, sample_prefix_mask, split_episodes


def read_config(path: str | Path) -> dict:
    config = json.loads(Path(path).read_text())
    p = config["policy"]
    if p["type"] != "openpi_pi05" or p["dtype"] != "bfloat16":
        raise ValueError("Expected openpi_pi05 with bfloat16.")
    if p["use_peft"] or p["freeze_vision_encoder"] or p["train_expert_only"] or config["peft"] is not None:
        raise ValueError("This entry point implements full-model training without LoRA.")
    if config["accelerator"]["gradient_accumulation"]["steps"] != 1:
        raise ValueError("This experiment expects gradient accumulation = 1.")
    if config["parallelism"] != {"dp_replicate": 4, "dp_shard": 1}:
        raise ValueError("This experiment expects four DDP ranks.")
    if config["wandb"]["enable"] or config["env_eval_freq"] != 0:
        raise ValueError("Only local loss logging and offline validation are supported here.")
    if not 0 <= p["rtc_training_max_delay"] < p["chunk_size"]:
        raise ValueError("RTC max delay must be smaller than the action horizon.")
    pose = config.get("pose_delta", {})
    expected = {"schema_version": 1, "stored_pose": "xyz_quat_wxyz_gripper_per_arm",
                "state_rotation": "rotation_6d_columns", "action_rotation": "rotvec",
                "action_dim": 14, "state_dim": 20, "position_unit": "m", "rotation_unit": "rad",
                "gripper": "absolute_m", "origin": "current_observation_for_entire_chunk"}
    if any(pose.get(key) != value for key, value in expected.items()) or pose.get("delta_frame") not in ("base", "tcp"):
        raise ValueError("Expected an explicit, supported pose_delta configuration.")
    if p["use_relative_actions"] or p["max_action_dim"] < max(pose["action_dim"], pose["state_dim"]):
        raise ValueError("Use the pose transform only, with adequate model padding width.")
    joint_root = Path(config["preparation"]["aligned_joint_root"]).resolve()
    dataset_root = Path(config["dataset"]["root"]).resolve()
    if dataset_root == joint_root or joint_root in dataset_root.parents:
        raise ValueError("Pose training requires its own prepared dataset.")
    output = Path(config["output_dir"]).resolve()
    if not output.name.startswith("openpi_pi05_dualarm_fold_towel_pose_delta_"):
        raise ValueError("Use a separate output directory named openpi_pi05_dualarm_fold_towel_pose_delta_*.")
    for value in (config["dataset"]["root"], p["pretrained_path"], p["text_tokenizer_name"]):
        if not Path(value).is_dir():
            raise FileNotFoundError(f"{value}. Prepare the pose dataset with --prepare-data first.")
    load_pose_stats(config)
    return config


def make_datasets(config):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    data = config["dataset"]
    meta = LeRobotDatasetMetadata(data["repo_id"], root=data["root"])
    train_ids, eval_ids = split_episodes(meta, data["eval_split"])
    kwargs = dict(
        repo_id=data["repo_id"], root=data["root"], video_backend=data["video_backend"],
        delta_timestamps={"action": [i / meta.fps for i in range(config["policy"]["chunk_size"])]},
    )
    train = LeRobotDataset(**kwargs, episodes=train_ids)
    validation = LeRobotDataset(**kwargs, episodes=eval_ids)
    # Same task-balanced first-frame selection as lerobot_train.py.
    tasks = validation.hf_dataset.data.column("task_index").to_numpy()
    per_task = max(1, config["max_eval_samples"] // len(set(tasks.tolist())))
    selected = []
    for task in sorted(set(tasks.tolist())):
        selected.extend(np.flatnonzero(tasks == task)[:per_task].tolist())
    return train, Subset(validation, selected), Collator(config, train.meta), train_ids, eval_ids


def loader(dataset, collator, config, sampler=None, *, evaluation=False):
    workers = config["num_workers"]
    return DataLoader(
        dataset, batch_size=config["batch_size"], sampler=sampler, shuffle=False,
        num_workers=workers, collate_fn=collator, pin_memory=True, drop_last=not evaluation,
        persistent_workers=workers > 0, prefetch_factor=4 if workers else None,
        multiprocessing_context="spawn" if workers else None,
        generator=torch.Generator().manual_seed(config["seed"]),
    )


def validate_weights(config):
    """Compare every key/shape on the meta device, without allocating the full model."""
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

    with torch.device("meta"):
        model = PI0Pytorch(model_config(config))
    expected = model.state_dict()
    with safe_open(str(Path(config["policy"]["pretrained_path"]) / "model.safetensors"), framework="pt") as f:
        available = set(f.keys())
        aliases = f.metadata() or {}
        for key, tensor in expected.items():
            stored = key if key in available else aliases.get(key)
            if stored not in available or tuple(f.get_slice(stored).get_shape()) != tuple(tensor.shape):
                raise ValueError(f"Pretrained key missing or shape mismatch: {key}")
        unexpected = available - expected.keys()
        if unexpected:
            raise ValueError(f"Unexpected pretrained keys: {sorted(unexpected)[:10]}")
    return len(expected)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state())


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state(state["cuda"])


def checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, config: dict,
               step: int, epoch: int, batch_offset: int, rank: int, split: dict) -> None:
    states = [None] * dist.get_world_size()
    dist.all_gather_object(states, rng_state())
    if rank == 0:
        target = Path(config["output_dir"]) / "checkpoints" / f"{step:06d}"
        staging = target.with_name(target.name + ".incomplete")
        staging.mkdir(parents=True, exist_ok=False)
        save_model(model, str(staging / "model.safetensors"))
        torch.save({"optimizer": optimizer.state_dict(), "step": step, "epoch": epoch,
                    "batch_offset": batch_offset, "rng": states}, staging / "training_state.pt")
        (staging / "train_config.json").write_text(json.dumps(config, indent=2) + "\n")
        (staging / "split.json").write_text(json.dumps(split, indent=2) + "\n")
        shutil.copy2(Path(config["dataset"]["root"]) / "meta/stats.json", staging / "dataset_stats.json")
        shutil.copy2(Path(config["dataset"]["root"]) / config["pose_delta"]["stats_file"], staging / "pose_delta_stats.json")
        shutil.copy2(Path(__file__).parent / "UPSTREAM.json", staging / "UPSTREAM.json")
        staging.rename(target)
        logging.info("Saved %s", target)
    dist.barrier()


@torch.no_grad()
def evaluate(model: torch.nn.Module, batches: DataLoader, device: torch.device, config: dict) -> float:
    saved = rng_state()
    model.eval()
    total, count = 0.0, 0
    try:
        torch.manual_seed(config["seed"] + 1)
        torch.cuda.manual_seed(config["seed"] + 1)
        for obs, actions in batches:
            obs = observation_to_device(obs, device)
            actions = actions.to(device)
            prefix = sample_prefix_mask(actions, config["policy"]["rtc_training_max_delay"])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = rtc_loss(model(obs, actions, prefix_mask=prefix), prefix, config["pose_delta"]["action_dim"])
            total += loss.item() * len(actions)
            count += len(actions)
    finally:
        model.train()
        restore_rng(saved)
    return total / count


def append_metric(output, filename, fields):
    path = Path(output) / filename
    new = not path.exists()
    with path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields))
        if new:
            writer.writeheader()
        writer.writerow(fields)


def train(config: dict, resume: Path | None) -> None:
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

    stats_path = Path(config["dataset"]["root"]) / config["pose_delta"]["stats_file"]
    if resume:
        checkpoint_path = Path(resume)
        previous = json.loads((checkpoint_path / "train_config.json").read_text())
        if previous != config:
            raise ValueError("Resume requires the unchanged pose checkpoint configuration.")
        if stats_path.read_bytes() != (checkpoint_path / "pose_delta_stats.json").read_bytes():
            raise ValueError("Pose normalization statistics changed since this checkpoint.")
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    if int(os.environ["WORLD_SIZE"]) != 4:
        raise ValueError("Launch this config with torchrun --nproc-per-node=4.")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    seed = config["seed"] + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    output = Path(config["output_dir"])
    if rank == 0:
        if not resume and output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Output is not empty: {output}. Use --resume or another output_dir.")
        output.mkdir(parents=True, exist_ok=True)
        (output / "train_config.json").write_text(json.dumps(config, indent=2) + "\n")
        shutil.copy2(stats_path, output / "pose_delta_stats.json")
        # Record the exact local source, including the RTC patch and v3 adapter.
        shutil.copytree(Path(__file__).parent, output / "source/training", dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copytree(ROOT / "src/openpi/models_pytorch", output / "source/models_pytorch",
                        dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
    dist.barrier()
    dataset, validation, collator, train_ids, eval_ids = make_datasets(config)
    split = {"train_episodes": train_ids, "eval_episodes": eval_ids, "eval_samples": len(validation)}
    if rank == 0:
        (output / "split.json").write_text(json.dumps(split, indent=2) + "\n")
        logging.info("Split: %d train episodes / %d eval episodes; %d validation frames",
                     len(train_ids), len(eval_ids), len(validation))
    sampler = DistributedSampler(dataset, num_replicas=4, rank=rank, seed=config["seed"], shuffle=True)
    batches = loader(dataset, collator, config, sampler)
    eval_batches = loader(validation, collator, config, evaluation=True) if rank == 0 else None
    model = PI0Pytorch(model_config(config))
    weights = Path(resume) if resume else Path(config["policy"]["pretrained_path"])
    load_model(model, str(weights / "model.safetensors"), strict=True)
    model.to(device)
    if config["policy"]["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
    model.train()
    p = config["policy"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=p["optimizer_lr"], betas=tuple(p["optimizer_betas"]),
                                  eps=p["optimizer_eps"], weight_decay=p["optimizer_weight_decay"])
    step, epoch, offset, saved_rng = 0, 0, 0, None
    if resume:
        previous = json.loads((weights / "train_config.json").read_text())
        if previous != config:
            raise ValueError("Resume must use the checkpoint's unchanged train_config.json.")
        saved = torch.load(weights / "training_state.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(saved["optimizer"])
        step, epoch, offset = saved["step"], saved["epoch"], saved["batch_offset"]
        saved_rng = saved["rng"][rank]
        del saved
        logging.info("Rank %d: resumed %s at step=%d/%d epoch=%d batch_offset=%d; next_lr=%.9g",
                     rank, weights, step, config["steps"], epoch, offset, learning_rate(step, p))
    ddp = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], find_unused_parameters=True, gradient_as_bucket_view=False,
    )
    running = torch.zeros((), device=device)
    window = 0
    started = time.monotonic()
    logging.info("Rank %d: official PI0Pytorch full model, 14D pose deltas (%s), 20D state, batch %d/rank, RTC delay %d",
                 rank, config["pose_delta"]["delta_frame"], config["batch_size"], p["rtc_training_max_delay"])
    while step < config["steps"]:
        sampler.set_epoch(epoch)
        for batch_index, (obs, actions) in enumerate(batches):
            if batch_index < offset:
                continue
            if saved_rng is not None:
                restore_rng(saved_rng)
                saved_rng = None
            obs = observation_to_device(obs, device)
            actions = actions.to(device, non_blocking=True)
            prefix = sample_prefix_mask(actions, p["rtc_training_max_delay"])
            lr = learning_rate(step, p)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = rtc_loss(ddp(obs, actions, prefix_mask=prefix), prefix, len(collator.names))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at step {step} on rank {rank}")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), p["optimizer_grad_clip_norm"], error_if_nonfinite=True)
            optimizer.step()
            step += 1
            running += loss.detach()
            window += 1
            if step % config["log_freq"] == 0 or step == config["steps"]:
                dist.all_reduce(running)
                value = running.item() / (window * 4)
                if rank == 0:
                    logging.info("step=%d loss=%.6f lr=%.9g grad_norm=%.4f elapsed=%.1fs", step, value, lr, norm.item(), time.monotonic()-started)
                    append_metric(output, "train_loss.csv", dict(step=step, loss=value, lr=lr, elapsed=time.monotonic()-started))
                running.zero_()
                window = 0
            if config["eval_steps"] and step % config["eval_steps"] == 0:
                dist.barrier()
                if rank == 0:
                    value = evaluate(model, eval_batches, device, config)
                    logging.info("step=%d eval_loss=%.6f", step, value)
                    append_metric(output, "eval_loss.csv", dict(step=step, loss=value, samples=len(validation)))
                dist.barrier()
            if step % config["save_freq"] == 0 or step == config["steps"]:
                checkpoint(model, optimizer, config, step, epoch, batch_index + 1, rank, split)
            if step >= config["steps"]:
                break
        epoch += 1
        offset = 0
    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", type=Path, help="Checkpoint directory, containing its train_config.json")
    parser.add_argument("--check-data", action="store_true", help="CPU-only: validate real data, tokenizer and pretrained shapes")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = read_config(args.config)
    if args.check_data:
        dataset, validation, collator, train_ids, eval_ids = make_datasets(config)
        obs, actions = collator([dataset[0], dataset[len(dataset)-1]])
        eval_obs, eval_actions = collator([validation[0]])
        assert actions.shape == (2, config["policy"]["chunk_size"], 32)
        assert torch.isfinite(actions).all() and torch.isfinite(eval_actions).all()
        assert torch.count_nonzero(actions[..., config["pose_delta"]["action_dim"]:]) == 0
        assert obs["state"].shape == (2, config["policy"]["max_action_dim"])
        assert obs["tokenized_prompt"].shape == (2, 200)
        shapes = validate_weights(config)
        print(json.dumps(dict(train_episodes=len(train_ids), eval_episodes=len(eval_ids),
                              train_frames=len(dataset), eval_samples=len(validation),
                              action_shape=list(actions.shape), cameras=list(obs["images"]),
                              real_action_dim=config["pose_delta"]["action_dim"],
                              real_state_dim=config["pose_delta"]["state_dim"],
                              pose_stats_sha256=hashlib.sha256((Path(config["dataset"]["root"]) / config["pose_delta"]["stats_file"]).read_bytes()).hexdigest(),
                              checked_pretrained_keys=shapes, cuda_initialized=torch.cuda.is_initialized()), indent=2))
        return
    if "RANK" not in os.environ:
        parser.error("Use the four-GPU bash launcher, or --check-data for CPU validation.")
    train(config, args.resume)


if __name__ == "__main__":
    main()
