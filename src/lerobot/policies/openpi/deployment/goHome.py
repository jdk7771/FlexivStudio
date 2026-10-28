#!/usr/bin/env python
"""Use the existing raised home routine with the independent OpenPI configuration."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import shared

if __name__ == "__main__":
    sys.argv[1:1] = ["--config", str(Path(__file__).with_name("deploy_config.json"))]
    runpy.run_path(str(shared.PI05_SCRIPTS / "goHome.py"), run_name="__main__")
