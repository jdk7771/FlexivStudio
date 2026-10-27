"""CPU regression checks for RTC semantics, per-token AdaRMS and baseline parity."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from runtime import learning_rate, rtc_loss, sample_prefix_mask, split_episodes


def test_rtc_mask_and_loss_exclude_prefix_and_padded_dimensions():
    actions = torch.zeros(4096, 50, 32)
    torch.manual_seed(1000)
    mask = sample_prefix_mask(actions, 8)
    assert set(mask.sum(1).tolist()) == set(range(9))
    assert torch.all(mask[:, 1:].int() <= mask[:, :-1].int())
    losses = torch.ones(2, 50, 32, requires_grad=True)
    prefix = torch.arange(50)[None, :] < torch.tensor([0, 8])[:, None]
    result = rtc_loss(losses, prefix, 16)
    result.backward()
    assert result.item() == 1
    assert torch.all(losses.grad[1, :8] == 0)
    assert torch.all(losses.grad[..., 16:] == 0)
    assert torch.all(losses.grad[0, :, :16] > 0)


def test_tokenwise_adarms_matches_individual_tokens():
    from transformers.models.gemma.modeling_gemma import GemmaRMSNorm

    norm = GemmaRMSNorm(8, cond_dim=8)
    nn.init.normal_(norm.dense.weight)
    x, times = torch.randn(2, 5, 8), torch.randn(2, 5, 8)
    out, gate = norm(x, times)
    for index in range(5):
        expected, expected_gate = norm(x[:, index:index+1], times[:, index])
        torch.testing.assert_close(out[:, index:index+1], expected)
        torch.testing.assert_close(gate[:, index:index+1], expected_gate)
    shared = torch.randn(2, 8)
    a, b = norm(x, shared)
    c, d = norm(x, shared[:, None].expand(-1, 5, -1))
    torch.testing.assert_close(a, c)
    torch.testing.assert_close(b.expand_as(d), d)


def test_real_forward_receives_clean_prefix_and_zero_time():
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

    # Exercise the actual upstream forward without allocating 3B parameters.
    model = PI0Pytorch.__new__(PI0Pytorch)
    nn.Module.__init__(model)
    model.pi05 = True
    model.config = SimpleNamespace(action_horizon=5, image_augmentation=False)
    model.gradient_checkpointing_enabled = False
    model._preprocess_observation = lambda obs, train: ([], [], None, None, None)
    model.embed_prefix = lambda *args: (torch.zeros(2, 1, 4), torch.ones(2, 1, dtype=torch.bool), torch.zeros(2, 1))
    recorded = {}

    def suffix(state, actions, times):
        recorded.update(actions=actions, times=times)
        return actions, torch.ones(2, 5, dtype=torch.bool), torch.zeros(2, 5), times

    model.embed_suffix = suffix

    class FakeExpert(nn.Module):
        def __init__(self):
            super().__init__()
            self.paligemma = SimpleNamespace(language_model=SimpleNamespace(
                layers=[SimpleNamespace(self_attn=SimpleNamespace(q_proj=nn.Linear(4, 4)))]))

        def forward(self, **kwargs):
            return (None, kwargs["inputs_embeds"][1]), None

    model.paligemma_with_expert = FakeExpert()
    model.action_out_proj = nn.Identity()
    actions, noise = torch.randn(2, 5, 4), torch.randn(2, 5, 4)
    times = torch.tensor([0.25, 0.75])
    prefix = torch.arange(5)[None] < torch.tensor([0, 3])[:, None]
    result = model(None, actions, noise, times, prefix_mask=prefix)
    torch.testing.assert_close(recorded["actions"][prefix], actions[prefix])
    assert torch.all(recorded["times"][prefix] == 0)
    mixed = times[:, None, None] * noise + (1-times[:, None, None]) * actions
    torch.testing.assert_close(recorded["actions"][~prefix], mixed[~prefix])
    assert result.shape == actions.shape
    without = model(None, actions, noise, times)
    zero = model(None, actions, noise, times, prefix_mask=torch.zeros_like(prefix))
    torch.testing.assert_close(without, zero)


def test_per_token_time_embeddings():
    from openpi.models_pytorch.pi0_pytorch import create_sinusoidal_pos_embedding

    time = torch.tensor([[0.0, 0.8], [0.0, 0.2]])
    kwargs = dict(dimension=8, min_period=0.004, max_period=4.0, device=torch.device("cpu"))
    tokens = create_sinusoidal_pos_embedding(time, **kwargs)
    flat = create_sinusoidal_pos_embedding(time.flatten(), **kwargs).reshape(2, 2, 8)
    torch.testing.assert_close(tokens, flat)


def test_official_model_tiny_forward_backward(monkeypatch):
    """Run real upstream vision, joint attention and expert layers with tiny widths on CPU."""
    import openpi.models_pytorch.gemma_pytorch as gemma
    import openpi.models_pytorch.pi0_pytorch as pi0

    original_vlm, original_expert = gemma.PaliGemmaForConditionalGeneration, gemma.GemmaForCausalLM

    def small_vlm(config):
        config.text_config.vocab_size = 512
        config._vocab_size = 512
        config.image_token_index = 511
        config.vision_config.hidden_size = 32
        config.vision_config.intermediate_size = 64
        config.vision_config.num_attention_heads = 4
        config.vision_config.num_hidden_layers = 1
        config.vision_config.projection_dim = 32
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
    obs = SimpleNamespace(images={k: torch.zeros(1, 3, 224, 224) for k in keys},
                          image_masks={k: torch.ones(1, dtype=torch.bool) for k in keys},
                          state=torch.zeros(1, 32), tokenized_prompt=torch.ones(1, 4, dtype=torch.long),
                          tokenized_prompt_mask=torch.ones(1, 4, dtype=torch.bool),
                          token_ar_mask=None, token_loss_mask=None)
    actions = torch.randn(1, 10, 32)
    prefix = torch.arange(10)[None] < 8
    loss = rtc_loss(model(obs, actions, prefix_mask=prefix), prefix, 16)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.action_in_proj.weight.grad is not None
    assert model.paligemma_with_expert.paligemma.vision_tower.vision_model.embeddings.patch_embedding.weight.grad is not None


def test_split_and_lr_match_reference():
    meta = SimpleNamespace(total_episodes=123, episodes=[{"tasks": ["fold towel"]}] * 123)
    train, validation = split_episodes(meta, 0.08536585365853659)
    assert train == list(range(112)) and validation == list(range(112, 123))
    config = json.loads((Path(__file__).parent / "configs/dualarm_fold_towel_rtc_full.json").read_text())
    from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig

    p = config["policy"]
    optimizer = torch.optim.AdamW([nn.Parameter(torch.zeros(1))], lr=p["optimizer_lr"])
    scheduler = CosineDecayWithWarmupSchedulerConfig(
        num_warmup_steps=1000, num_decay_steps=50000, peak_lr=1e-5, decay_lr=2.5e-6
    ).build(optimizer, num_training_steps=50000)
    for step in (0, 1, 999, 1000, 25000, 50000):
        assert learning_rate(step, p) == pytest.approx(p["optimizer_lr"] * scheduler.lr_lambdas[0](step))
