"""Independent TCP pose-delta adapter and RTC objective for official OpenPI."""

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import default_collate

from pose_math import ACTION_NAMES, POSE_NAMES, STATE_NAMES, pose_delta, state_features


def load_pose_stats(config: dict) -> dict:
    path = Path(config["dataset"]["root"]) / config["pose_delta"]["stats_file"]
    stats = json.loads(path.read_text())
    if stats["pose_delta"] != config["pose_delta"] or stats["chunk_size"] != config["policy"]["chunk_size"]:
        raise ValueError("Pose statistics do not match the configured transform/horizon; prepare a new dataset.")
    if stats["state_names"] != STATE_NAMES or stats["action_names"] != ACTION_NAMES:
        raise ValueError("Pose statistics have incompatible feature ordering.")
    for key, width in (("observation.state", 20), ("action", 14)):
        lo, hi = (np.asarray(stats[key][q], dtype=np.float64) for q in ("q01", "q99"))
        if lo.shape != (width,) or hi.shape != (width,) or not np.isfinite([lo, hi]).all() or (hi < lo).any():
            raise ValueError(f"Invalid {key} normalization statistics.")
    return stats

def split_episodes(meta, fraction):
    by_task = {}
    for index in range(meta.total_episodes):
        tasks = meta.episodes[index]["tasks"]
        by_task.setdefault(tasks[0] if tasks else "", []).append(index)
    train, validation = [], []
    for episodes in by_task.values():
        count = math.ceil(len(episodes) * fraction)
        boundary = len(episodes) - count
        train.extend(episodes[:boundary])
        validation.extend(episodes[boundary:])
    if not train or not validation:
        raise ValueError("Expected nonempty training and validation episode splits.")
    return train, validation


def sample_prefix_mask(actions, max_delay):
    if not 0 <= max_delay < actions.shape[1]:
        raise ValueError("RTC delay must be in [0, action_horizon).")
    delays = torch.randint(max_delay + 1, (actions.shape[0],), device=actions.device)
    return torch.arange(actions.shape[1], device=actions.device)[None, :] < delays[:, None]


def rtc_loss(losses, prefix_mask, action_dim):
    """Match LeRobot: average only real action dimensions and the predicted postfix."""
    losses = losses[..., :action_dim]
    weights = (~prefix_mask)[..., None].expand_as(losses)
    return (losses * weights).sum() / weights.sum().clamp_min(1)


def learning_rate(step, policy):
    # The reference LeRobot schedule uses the absolute step in its cosine phase.
    peak = policy["optimizer_lr"]
    warmup = policy["scheduler_warmup_steps"]
    decay = policy["scheduler_decay_steps"]
    if step < warmup:
        initial = peak / (warmup + 1)
        return initial + (peak - initial) * step / warmup
    progress = min(step, decay) / decay
    return policy["scheduler_decay_lr"] + (peak - policy["scheduler_decay_lr"]) * (
        1 + math.cos(math.pi * progress)
    ) / 2


class Collator:
    """Transform absolute TCP chunks into normalized 14D deltas; encode 20D state."""

    def __init__(self, config: dict, meta):
        self.policy = config["policy"]
        self.frame = config["pose_delta"]["delta_frame"]
        self.stats = load_pose_stats(config)
        self.names = ACTION_NAMES
        self.cameras = ["observation.images.image", "observation.images.image2", "observation.images.image3"]
        if set(meta.camera_keys) != set(self.cameras):
            raise ValueError("Expected base, left wrist and right wrist camera keys.")
        for key in ("observation.state", "action"):
            if meta.features[key]["names"] != POSE_NAMES or list(meta.features[key]["shape"]) != [16]:
                raise ValueError("Expected absolute TCP poses; a joint dataset cannot be used here.")
        train_ids, eval_ids = split_episodes(meta, config["dataset"]["eval_split"])
        if train_ids != self.stats["train_episodes"] or eval_ids != self.stats["eval_episodes"]:
            raise ValueError("Statistics were fitted to a different train/eval split.")
        self.tokenizer = None

    def normalize(self, value: torch.Tensor, key: str) -> torch.Tensor:
        lo = torch.as_tensor(self.stats[key]["q01"], dtype=torch.float32)
        hi = torch.as_tensor(self.stats[key]["q99"], dtype=torch.float32)
        denom = hi - lo
        # Constant dimensions map to zero and decode to their training constant.
        normalized = 2 * (value - lo) / denom.clamp_min(1e-6) - 1
        return torch.where(denom < 1e-6, torch.zeros_like(normalized), normalized)

    def denormalize_actions(self, value: np.ndarray) -> np.ndarray:
        lo = np.asarray(self.stats["action"]["q01"], dtype=np.float32)
        hi = np.asarray(self.stats["action"]["q99"], dtype=np.float32)
        value = np.asarray(value, dtype=np.float32)[..., :len(self.names)]
        return np.where(hi - lo < 1e-6, lo, (value + 1) * (hi - lo) / 2 + lo)

    def __call__(self, samples: list[dict]) -> tuple[dict, torch.Tensor]:
        from transformers import AutoTokenizer

        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.policy["text_tokenizer_name"], local_files_only=True, padding_side="right"
            )
        batch = default_collate(samples)
        poses = batch["observation.state"].float().numpy()
        targets = batch["action"].float().numpy()
        actions = torch.from_numpy(pose_delta(poses[:, None, :], targets, self.frame))
        state = torch.from_numpy(state_features(poses))
        actions = self.normalize(actions, "action")
        state = self.normalize(state, "observation.state")
        discrete = np.digitize(state.numpy(), np.linspace(-1, 1, 257)[:-1]) - 1
        prompts = []
        for task, values in zip(batch["task"], discrete, strict=True):
            task = task.strip().replace("_", " ").replace("\n", " ")
            prompts.append(f"Task: {task}, State: {' '.join(map(str, values))};\nAction: ")
        tokens = self.tokenizer(
            prompts, max_length=self.policy["tokenizer_max_length"], padding="max_length",
            truncation=True, return_tensors="pt",
        )
        image_names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        images = {}
        for key, target in zip(self.cameras, image_names, strict=True):
            value = batch[key]
            images[target] = value.float() / 127.5 - 1 if value.dtype == torch.uint8 else value * 2 - 1
        width = self.policy["max_action_dim"]
        return {
            "images": images,
            "image_masks": {key: torch.ones(len(samples), dtype=torch.bool) for key in images},
            "state": torch.nn.functional.pad(state, (0, width - state.shape[-1])),
            "tokenized_prompt": tokens.input_ids,
            "tokenized_prompt_mask": tokens.attention_mask.bool(),
        }, torch.nn.functional.pad(actions, (0, width - actions.shape[-1]))


def observation_to_device(batch, device):
    result = {}
    for key, value in batch.items():
        result[key] = (
            {name: tensor.to(device, non_blocking=True) for name, tensor in value.items()}
            if isinstance(value, dict) else value.to(device, non_blocking=True)
        )
    return SimpleNamespace(**result, token_ar_mask=None, token_loss_mask=None)


def model_config(config):
    p = config["policy"]
    return SimpleNamespace(
        pi05=True, dtype=p["dtype"], action_dim=p["max_action_dim"], action_horizon=p["chunk_size"],
        paligemma_variant="gemma_2b", action_expert_variant="gemma_300m", pytorch_compile_mode=None,
        image_augmentation=p["image_augmentation"], rtc_training_max_delay=p["rtc_training_max_delay"],
    )
