"""LeRobot v3 data adapter and RTC objective for the official openpi PyTorch model."""

import math
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import default_collate


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
    """Use the reference dataset statistics, prompt format, camera order and tokenizer."""

    def __init__(self, config, meta):
        self.policy = config["policy"]
        self.stats = meta.stats
        self.names = meta.features["action"]["names"]
        self.cameras = list(meta.camera_keys)
        if len(self.cameras) != 3 or len(self.names) != 16:
            raise ValueError("This experiment requires the three-camera, 16-dimensional dual-arm dataset.")
        self.tokenizer = None

    def normalize(self, value, key):
        lo = torch.as_tensor(self.stats[key]["q01"], dtype=torch.float32)
        hi = torch.as_tensor(self.stats[key]["q99"], dtype=torch.float32)
        denom = hi - lo
        denom = torch.where(denom == 0, 1e-8, denom)
        return 2 * (value - lo) / denom - 1

    def __call__(self, samples):
        from transformers import AutoTokenizer

        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.policy["text_tokenizer_name"], local_files_only=True, padding_side="right"
            )
        batch = default_collate(samples)
        state = batch["observation.state"].float()
        actions = batch["action"].float().clone()
        if self.policy["use_relative_actions"]:
            relative = [
                not any(exclusion in name for exclusion in self.policy["relative_exclude_joints"])
                for name in self.names
            ]
            actions[..., relative] -= state[:, None, relative]
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
