#!/usr/bin/env python3
"""Compare PI0.5 checkpoints on held-out demonstration frames, without hardware."""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_train_eval_datasets
from lerobot.policies.pi05 import PI05Policy  # Registers pi05 for saved config parsing.
from lerobot.policies.factory import make_pre_post_processors
from lerobot.scripts.lerobot_train import _preprocess_dataset_batch
from lerobot.utils.constants import ACTION, OBS_STATE


def select_indices(dataset, per_episode: int, horizon: int) -> list[int]:
    episode_ids = dataset.hf_dataset.data.column("episode_index").to_numpy()
    selected = []
    for episode in dataset.episodes:
        indices = np.flatnonzero(episode_ids == episode)
        valid = indices[: max(0, len(indices) - horizon + 1)]
        if len(valid) == 0:
            raise ValueError(f"Held-out episode {episode} is shorter than the {horizon}-step horizon")
        positions = np.linspace(0, len(valid) - 1, min(per_episode, len(valid)), dtype=int)
        selected.extend(valid[positions].tolist())
    return selected


def summarize(errors: list[dict], field: str) -> dict[str, float]:
    return {
        "mean": float(np.mean([item[field] for item in errors])),
        "median": float(np.median([item[field] for item in errors])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoints", nargs="+", default=["007500", "050000"])
    parser.add_argument("--per-episode", type=int, default=6)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.per_episode < 1 or args.horizon < 1:
        parser.error("--per-episode and --horizon must be positive")
    if not torch.cuda.is_available():
        parser.error("CUDA is required for PI0.5 inference")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    checkpoint_dirs = [args.run_dir / "checkpoints" / step / "pretrained_model" for step in args.checkpoints]
    for path in checkpoint_dirs:
        if not (path / "model.safetensors").is_file():
            raise FileNotFoundError(f"Missing checkpoint weights: {path}")

    train_cfg = TrainPipelineConfig.from_pretrained(checkpoint_dirs[-1])
    _, dataset = make_train_eval_datasets(train_cfg)
    if dataset is None:
        raise ValueError("The saved training config has no held-out evaluation split")
    indices = select_indices(dataset, args.per_episode, args.horizon)
    samples = [dataset[index] for index in indices]
    logging.info("Loaded %d held-out frames from %d episodes", len(samples), len(dataset.episodes))

    action_names = train_cfg.policy.action_feature_names
    joint_dims = [index for index, name in enumerate(action_names) if "gripper" not in name.lower()]
    gripper_dims = [index for index, name in enumerate(action_names) if "gripper" in name.lower()]
    if len(joint_dims) != 14 or len(gripper_dims) != 2:
        raise ValueError(f"Unexpected action dimensions: {action_names}")

    baseline = []
    for index, sample in zip(indices, samples, strict=True):
        target = sample[ACTION][: args.horizon].float()
        state = sample[OBS_STATE].float()
        repeated_state = state.expand_as(target)
        difference = (repeated_state - target).abs()
        baseline.append(
            {
                "dataset_index": index,
                "episode": int(sample["episode_index"]),
                "joint_first_mae": float(difference[0, joint_dims].mean()),
                "joint_horizon_mae": float(difference[:, joint_dims].mean()),
                "gripper_first_mae": float(difference[0, gripper_dims].mean()),
                "gripper_horizon_mae": float(difference[:, gripper_dims].mean()),
            }
        )

    report = {
        "dataset": str(train_cfg.dataset.root),
        "eval_episodes": list(dataset.episodes),
        "sample_count": len(samples),
        "samples_per_episode": args.per_episode,
        "horizon": args.horizon,
        "seed": args.seed,
        "metrics": "Mean absolute action error against demonstrations; joint values in dataset joint units, grippers in dataset gripper units.",
        "baseline_hold_state": {
            key: summarize(baseline, key)
            for key in ("joint_first_mae", "joint_horizon_mae", "gripper_first_mae", "gripper_horizon_mae")
        },
        "checkpoints": {},
    }

    device = torch.device("cuda:0")
    for checkpoint_dir in checkpoint_dirs:
        logging.info("Loading %s", checkpoint_dir)
        policy = PI05Policy.from_pretrained(checkpoint_dir, strict=True).to(device).eval()
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=checkpoint_dir,
            preprocessor_overrides={"device_processor": {"device": str(device)}},
        )
        errors = []
        generator = torch.Generator(device=device)
        with torch.inference_mode():
            for position, (index, sample) in enumerate(zip(indices, samples, strict=True)):
                observations = {key: value for key, value in sample.items() if key != ACTION}
                batch = _preprocess_dataset_batch(
                    observations, dataset.meta.camera_keys, train_cfg.rename_map, preprocessor
                )
                generator.manual_seed(args.seed + index)
                noise = torch.randn(
                    (1, policy.config.chunk_size, policy.config.max_action_dim),
                    device=device,
                    dtype=torch.float32,
                    generator=generator,
                )
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    prediction = policy.predict_action_chunk(batch, noise=noise)
                absolute_actions = postprocessor(prediction).detach().cpu().float()
                target = sample[ACTION][: args.horizon].float()
                difference = (absolute_actions[0, : args.horizon] - target).abs()
                errors.append(
                    {
                        "dataset_index": index,
                        "episode": int(sample["episode_index"]),
                        "joint_first_mae": float(difference[0, joint_dims].mean()),
                        "joint_horizon_mae": float(difference[:, joint_dims].mean()),
                        "gripper_first_mae": float(difference[0, gripper_dims].mean()),
                        "gripper_horizon_mae": float(difference[:, gripper_dims].mean()),
                    }
                )
                if (position + 1) % 10 == 0 or position + 1 == len(samples):
                    logging.info("%s: %d/%d frames", checkpoint_dir.parent.name, position + 1, len(samples))

        report["checkpoints"][checkpoint_dir.parent.name] = {
            "summary": {
                key: summarize(errors, key)
                for key in ("joint_first_mae", "joint_horizon_mae", "gripper_first_mae", "gripper_horizon_mae")
            },
            "samples": errors,
        }
        del policy, preprocessor, postprocessor
        torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    logging.info("Saved %s", args.output)


if __name__ == "__main__":
    main()
