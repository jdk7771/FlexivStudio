"""Official OpenPI PyTorch inference with training-identical transforms and RTC clamping."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import shared  # noqa: F401

# isort: split

import numpy as np
import torch
from flexiv_deploy_contract import ACTION_NAMES, CAMERA_NAMES

JOINTS = np.r_[0:7, 8:15]


class Transforms:
    def __init__(self, train: dict, stats: dict, tokenizer: Any) -> None:
        self.p = train["policy"]
        self.tokenizer = tokenizer
        self.stats = {}
        for key in ("action", "observation.state"):
            lo = np.asarray(stats[key]["q01"], dtype=np.float32)
            hi = np.asarray(stats[key]["q99"], dtype=np.float32)
            self.stats[key] = lo, np.where(hi == lo, np.float32(1e-8), hi - lo)

    def normalize(self, value: np.ndarray, key: str) -> np.ndarray:
        lo, width = self.stats[key]
        return 2 * (value - lo) / width - 1

    def observation(self, obs: dict, task: str, device: str) -> tuple[Any, np.ndarray]:
        state = np.asarray([obs[n] for n in ACTION_NAMES], dtype=np.float32)
        if state.shape != (16,) or not np.isfinite(state).all():
            raise ValueError("Expected 16 finite measured state channels")
        normalized = self.normalize(state, "observation.state")
        # Exact Collator prompt, quantile normalization and 16-dimensional state discretization.
        values = np.asarray(np.digitize(normalized, np.linspace(-1, 1, 257)[:-1]) - 1)
        task = task.strip().replace("_", " ").replace("\n", " ")
        prompt = f"Task: {task}, State: {' '.join(map(str, values))};\nAction: "
        tokens = self.tokenizer(
            [prompt],
            max_length=self.p["tokenizer_max_length"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        images = {}
        for source, target in zip(
            CAMERA_NAMES, ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"), strict=True
        ):
            image = np.asarray(obs[source])
            if image.shape != (224, 224, 3) or image.dtype != np.uint8:
                raise ValueError(f"{source} must be letterboxed RGB uint8 (224,224,3)")
            images[target] = (
                torch.from_numpy(image.copy()).permute(2, 0, 1)[None].to(device).float() / 127.5 - 1
            )
        result = SimpleNamespace(
            images=images,
            image_masks={k: torch.ones(1, dtype=torch.bool, device=device) for k in images},
            state=torch.from_numpy(np.pad(normalized, (0, 16)))[None].to(device),
            tokenized_prompt=tokens.input_ids.to(device),
            tokenized_prompt_mask=tokens.attention_mask.bool().to(device),
            token_ar_mask=None,
            token_loss_mask=None,
        )
        return result, state

    def encode_actions(self, absolute: np.ndarray, anchor: np.ndarray) -> np.ndarray:
        values = np.asarray(absolute, dtype=np.float32).copy()
        if values.ndim != 2 or values.shape[1] != 16 or not np.isfinite(values).all():
            raise ValueError("RTC prefix must contain finite absolute 16-dimensional targets")
        values[:, JOINTS] -= anchor[JOINTS]
        return np.pad(self.normalize(values, "action"), ((0, 0), (0, 16)))

    def decode_actions(self, normalized: np.ndarray, anchor: np.ndarray) -> np.ndarray:
        lo, width = self.stats["action"]
        values = (normalized[:, :16] + 1) / 2 * width + lo
        values[:, JOINTS] += anchor[JOINTS]
        if not np.isfinite(values).all():
            raise ValueError("Model returned non-finite actions")
        return values


@torch.inference_mode()
def sample_rtc(
    model: Any,
    device: str,
    observation: Any,
    prefix: torch.Tensor,
    num_steps: int,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mirror upstream sampling; clean hard prefix has time=0 at EVERY denoising step."""
    from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

    batch = observation.state.shape[0]
    shape = (batch, model.config.action_horizon, model.config.action_dim)
    count = prefix.shape[1]
    if not 0 <= count < shape[1] or prefix.shape != (batch, count, shape[2]) or num_steps <= 0:
        raise ValueError("Invalid RTC prefix or denoising step count")
    images, masks, tokens, token_masks, state = model._preprocess_observation(observation, train=False)
    emb, pad, att = model.embed_prefix(images, masks, tokens, token_masks)
    attention = model._prepare_attention_masks_4d(make_att_2d_masks(pad, att))
    model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
    _, cache = model.paligemma_with_expert.forward(
        attention_mask=attention,
        position_ids=torch.cumsum(pad, dim=1) - 1,
        past_key_values=None,
        inputs_embeds=[emb, None],
        use_cache=True,
    )
    x = model.sample_noise(shape, device) if noise is None else noise.clone()
    # Match upstream float32 accumulation exactly, including its rounding at each Euler step.
    dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
    diffusion_time = torch.tensor(1.0, dtype=torch.float32, device=device)
    for _ in range(num_steps):
        if count:
            x[:, :count] = prefix
            times = diffusion_time.expand(batch, shape[1]).clone()
            times[:, :count] = 0
        else:
            times = diffusion_time.expand(batch)
        velocity = model.denoise_step(state, pad, cache, x, times)
        x = x + dt * velocity
        if count:
            x[:, :count] = prefix
        diffusion_time += dt
    return x


class OpenPiPolicy:
    def __init__(self, cfg: dict, description: dict) -> None:
        from importlib.metadata import version
        from pathlib import Path

        from safetensors.torch import load_model
        from transformers import AutoTokenizer
        from transformers.modeling_utils import no_init_weights

        if version("transformers") != "4.53.2":
            raise RuntimeError("Use bash deployment/setup.sh and deployment/run.sh (transformers 4.53.2)")
        if not cfg["device"].startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError("OpenPI deployment requires CUDA")
        torch.cuda.set_device(torch.device(cfg["device"]))
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("OpenPI requires a BF16-capable GPU")
        from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

        p = description["train"]["policy"]
        model_cfg = SimpleNamespace(
            pi05=True,
            dtype=p["dtype"],
            action_dim=32,
            action_horizon=p["chunk_size"],
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
            pytorch_compile_mode=None,
            image_augmentation=False,
            rtc_training_max_delay=p["rtc_training_max_delay"],
        )
        with no_init_weights():
            self.model = PI0Pytorch(model_cfg)
        # no_init_weights skips HF post_init/tie_weights; restore the alias saved by safetensors.
        self.model.paligemma_with_expert.paligemma.tie_weights()
        load_model(self.model, str(Path(cfg["checkpoint"]) / "model.safetensors"), strict=True, device="cpu")
        self.model.to(cfg["device"]).eval()
        self.model.gradient_checkpointing_disable()
        self.cfg = cfg
        # The local tokenizer was re-saved by transformers 5 (list-valued extra_special_tokens).
        # Restore the v4 mapping in memory; keep the original tokenizer files untouched.
        tokenizer = AutoTokenizer.from_pretrained(
            cfg["tokenizer"],
            local_files_only=True,
            padding_side="right",
            extra_special_tokens={"image_token": "<image>"},
        )
        self.transforms = Transforms(description["train"], description["stats"], tokenizer)

    def predict(self, obs: dict, prefix: np.ndarray | None = None) -> np.ndarray:
        device = self.cfg["device"]
        observation, anchor = self.transforms.observation(obs, self.cfg["task"], device)
        absolute_prefix = np.empty((0, 16), dtype=np.float32) if prefix is None else prefix
        encoded = self.transforms.encode_actions(absolute_prefix, anchor)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            actions = sample_rtc(
                self.model,
                device,
                observation,
                torch.from_numpy(encoded)[None].to(device),
                self.cfg["num_inference_steps"],
            )
        targets = self.transforms.decode_actions(actions[0].float().cpu().numpy(), anchor)
        hw = self.cfg["hardware"]
        targets[:, [7, 15]] = np.clip(targets[:, [7, 15]], hw["gripper_min_m"], hw["gripper_max_m"])
        # Avoid quantile roundoff on the already committed absolute prefix.
        targets[: len(absolute_prefix)] = absolute_prefix
        return targets
