"""Compare RTC directly with this workspace's LeRobot implementation, CPU only.

Load the original function/class AST nodes without importing LeRobot's transformers-5
policy into openpi's transformers-4 environment. No reference formula is reimplemented.
"""

import ast
import enum
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import Tensor, nn

from runtime import rtc_loss, sample_prefix_mask


REFERENCE = Path(__file__).resolve().parents[2]


def reference_symbols(relative_path, names, **dependencies):
    path = REFERENCE / relative_path
    tree = ast.parse(path.read_text(), filename=str(path))
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    assert {node.name for node in nodes} == set(names)
    namespace = dict(torch=torch, Tensor=Tensor, nn=nn, enum=enum, math=math, **dependencies)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in names})


@pytest.mark.parametrize("max_delay", [1, 8, 49])
def test_sample_mask_matches_local_lerobot_for_same_rng(max_delay):
    ref = reference_symbols("pi05/modeling_pi05.py", ["_sample_training_rtc_prefix_mask"])
    actions = torch.zeros(128, 50, 32)
    torch.manual_seed(1000)
    expected = ref._sample_training_rtc_prefix_mask(128, 50, max_delay, actions.device)
    expected_rng = torch.get_rng_state()
    torch.manual_seed(1000)
    actual = sample_prefix_mask(actions, max_delay)
    assert torch.equal(actual, expected)
    assert torch.equal(torch.get_rng_state(), expected_rng)


@pytest.mark.parametrize("delays", [[0, 0, 0], [0, 1, 8], [8, 8, 8], [0, 8, 49]])
def test_real_openpi_forward_inputs_targets_loss_and_gradients_match_reference(delays):
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

    flow = reference_symbols("common/flow_matching.py", ["FlowConvention", "make_flow_matching_inputs"])
    reducer = reference_symbols("pi05/modeling_pi05.py", ["_reduce_training_rtc_loss"])
    torch.manual_seed(52)
    actions, noise = torch.randn(3, 50, 32), torch.randn(3, 50, 32)
    time = torch.tensor([0.001, 0.4, 0.999])
    prefix = torch.arange(50)[None] < torch.tensor(delays)[:, None]
    expected_x, expected_target, expected_time = flow.make_flow_matching_inputs(
        actions, noise, time, convention=flow.FlowConvention.NOISE_AT_ONE, prefix_mask=prefix
    )
    # Execute the actual openpi forward; replace only expensive transformer execution.
    model = PI0Pytorch.__new__(PI0Pytorch)
    nn.Module.__init__(model)
    model.pi05 = True
    model.config = SimpleNamespace(action_horizon=50, image_augmentation=False)
    model.gradient_checkpointing_enabled = False
    model._preprocess_observation = lambda obs, train: ([], [], None, None, None)
    model.embed_prefix = lambda *args: (
        torch.zeros(3, 1, 32), torch.ones(3, 1, dtype=torch.bool), torch.zeros(3, 1)
    )

    def suffix(state, x, timestep):
        torch.testing.assert_close(x, expected_x, rtol=0, atol=0)
        torch.testing.assert_close(timestep, expected_time, rtol=0, atol=0)
        return x, torch.ones(3, 50, dtype=torch.bool), torch.zeros(3, 50), timestep

    model.embed_suffix = suffix

    class Expert(nn.Module):
        def __init__(self):
            super().__init__()
            self.paligemma = SimpleNamespace(language_model=SimpleNamespace(
                layers=[SimpleNamespace(self_attn=SimpleNamespace(q_proj=nn.Linear(32, 32)))]
            ))

        def forward(self, **kwargs):
            return (None, kwargs["inputs_embeds"][1]), None

    model.paligemma_with_expert = Expert()
    model.action_out_proj = nn.Linear(32, 32)
    losses = model(None, actions, noise=noise, time=time, prefix_mask=prefix)
    actual = rtc_loss(losses, prefix, 16)
    prediction = model.action_out_proj(expected_x)
    expected_losses = (prediction - expected_target).square()
    expected = reducer._reduce_training_rtc_loss(expected_losses[..., :16], prefix, "mean")
    torch.testing.assert_close(losses, expected_losses, rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    params = tuple(model.action_out_proj.parameters())
    grad_actual = torch.autograd.grad(actual, params)
    grad_expected = torch.autograd.grad(expected, params)
    for actual_grad, expected_grad in zip(grad_actual, grad_expected, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_adarms_outputs_and_gradients_match_local_lerobot(dtype):
    from transformers.models.gemma.modeling_gemma import GemmaRMSNorm

    ref = reference_symbols("pi_gemma.py", ["PiGemmaRMSNorm"])
    torch.manual_seed(45)
    original = ref.PiGemmaRMSNorm(16, cond_dim=16)
    nn.init.normal_(original.dense.weight, std=0.2)
    patched = GemmaRMSNorm(16, cond_dim=16)
    patched.load_state_dict(original.state_dict(), strict=True)
    x = torch.randn(3, 50, 16).to(dtype).requires_grad_()
    condition = torch.randn(3, 50, 16).requires_grad_()
    a, gate_a = original(x, condition)
    b, gate_b = patched(x, condition)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(gate_a, gate_b, rtol=0, atol=0)
    grad_a = torch.autograd.grad(a.float().square().mean() + gate_a.float().square().mean(), (x, condition))
    grad_b = torch.autograd.grad(b.float().square().mean() + gate_b.float().square().mean(), (x, condition))
    for first, second in zip(grad_a, grad_b, strict=True):
        torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_time_embeddings_and_attention_mask_match_reference():
    from openpi.models_pytorch.pi0_pytorch import create_sinusoidal_pos_embedding, get_safe_dtype, make_att_2d_masks

    ref = reference_symbols("common/vla_utils.py", ["create_sinusoidal_pos_embedding", "make_att_2d_masks"],
                            get_safe_dtype=get_safe_dtype)
    times = torch.rand(3, 50)
    times[0, :8] = 0
    args = dict(dimension=1024, min_period=0.004, max_period=4.0, device=times.device)
    torch.testing.assert_close(create_sinusoidal_pos_embedding(times, **args),
                               ref.create_sinusoidal_pos_embedding(times, **args), rtol=0, atol=0)
    # Prefix images/text and bidirectional action block; RTC does not alter attention.
    padding = torch.ones(3, 60, dtype=torch.bool)
    padding[:, 8:10] = False
    attention = torch.zeros(3, 60, dtype=torch.bool)
    attention[:, 10] = True
    assert torch.equal(make_att_2d_masks(padding, attention), ref.make_att_2d_masks(padding, attention))


def test_running_source_snapshot_and_installed_patch_match_reviewed_code():
    import transformers

    root = Path(__file__).resolve().parents[1]
    source = Path("/mnt/work/dakun/outputs/openpi_pi05_dualarm_fold_towel_rtc_full/source")
    for name in ("runtime.py", "train.py"):
        assert (root / "training" / name).read_bytes() == (source / "training" / name).read_bytes()
    assert (root / "src/openpi/models_pytorch/pi0_pytorch.py").read_bytes() == (
        source / "models_pytorch/pi0_pytorch.py"
    ).read_bytes()
    relative = Path("models/gemma/modeling_gemma.py")
    replacement = root / "src/openpi/models_pytorch/transformers_replace" / relative
    installed = Path(transformers.__file__).parent / relative
    assert replacement.read_bytes() == installed.read_bytes()
