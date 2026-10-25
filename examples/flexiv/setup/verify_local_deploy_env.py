"""Verify imports and a tiny CUDA BF16 operation; never load a policy or connect hardware."""

from importlib.metadata import version

import flexivrdk
import pyrealsense2
import torch
from peft import PeftModel

from lerobot.inference import RTCInferenceEngine
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.rtc.configuration_rtc import RTCConfig


def main() -> None:
    for name in ("torch", "torchvision", "transformers", "peft", "lerobot", "flexivrdk", "pyrealsense2"):
        print(f"{name}: {version(name)}")
    if version("flexivrdk") != "1.9.2":
        raise RuntimeError("This robot software (3.11.1) requires flexivrdk 1.9.2")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this environment")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 is not supported")
    # Exercise compiled CUDA kernels on the actual card, without a model forward pass.
    sample = torch.eye(16, device="cuda:0", dtype=torch.bfloat16)
    assert float((sample @ sample).float().sum().item()) == 16.0
    rtc = RTCConfig(enabled=True, mode="trained", execution_horizon=10)
    cfg = PI05Config(device="cuda:0", dtype=torch.bfloat16, rtc_training_max_delay=8, rtc_config=rtc)
    assert cfg.rtc_config.mode == "trained"
    assert callable(PI05Policy) and callable(RTCInferenceEngine) and callable(PeftModel)
    assert hasattr(flexivrdk.Robot, "SendJointPosition")
    assert hasattr(pyrealsense2, "pipeline")
    print(f"GPU: {torch.cuda.get_device_name(0)}; CUDA wheel: {torch.version.cuda}; BF16 kernel: PASS")
    print("PI05/trained RTC/LoRA/RDK/RealSense imports: PASS")
    print("No model was instantiated and no robot/camera was connected.")


if __name__ == "__main__":
    main()
