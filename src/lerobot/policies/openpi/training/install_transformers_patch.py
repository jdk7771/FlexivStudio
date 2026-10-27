"""Install upstream transformers replacements plus token-wise RTC AdaRMS support."""

from importlib.metadata import version
from pathlib import Path
import sys

import transformers

if version("transformers") != "4.53.2":
    raise SystemExit("This upstream snapshot requires transformers==4.53.2.")
target = Path(transformers.__file__).resolve().parent
if not target.is_relative_to(Path(sys.prefix).resolve()) or sys.prefix == sys.base_prefix:
    raise SystemExit("Refusing to patch the shared/base environment; use openpi/.venv/bin/python.")
source = Path(__file__).resolve().parents[1] / "src/openpi/models_pytorch/transformers_replace"
for file in source.rglob("*.py"):
    destination = target / file.relative_to(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Atomic replacement also breaks uv hardlinks, leaving its cache untouched.
    staging = destination.with_suffix(".openpi.tmp")
    staging.write_bytes(file.read_bytes())
    staging.replace(destination)
print(f"Installed openpi transformers replacements in {target}")
