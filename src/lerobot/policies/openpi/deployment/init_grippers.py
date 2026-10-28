#!/usr/bin/env python
"""Initialize grippers separately, using the independent OpenPI configuration."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import shared

if __name__ == "__main__":
    sys.argv[1:1] = ["--config", str(Path(__file__).with_name("deploy_config.json"))]
    runpy.run_path(str(shared.PI05_DEPLOYMENT / "init_grippers.py"), run_name="__main__")
