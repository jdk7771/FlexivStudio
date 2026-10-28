"""Reuse the existing hardware/home helpers without editing the PI05 deployment."""

from __future__ import annotations

import os
import sys
from pathlib import Path

OPENPI_ROOT = Path(__file__).resolve().parents[1]
PI05_DEPLOYMENT = OPENPI_ROOT.parent / "pi05" / "deployment"
# Existing hardware modules are standalone scripts with sibling imports.
for directory in (PI05_DEPLOYMENT, OPENPI_ROOT / "src"):
    if str(directory) not in sys.path:
        sys.path.append(str(directory))
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
