#!/usr/bin/env python3
"""
nki-opt-models-v2/knob_guide.py
================================

Auto-generates ``KNOB_GUIDE.md`` from each model's ``KNOB_NAMES`` dict and
prints an empty ``knobs.json`` template (one entry per discovered model) to
stdout.

Usage:
    python knob_guide.py                     # print knobs.json template
    python knob_guide.py --write-guide       # regenerate KNOB_GUIDE.md
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys

_DIR = pathlib.Path(__file__).parent
sys.path.insert(0, str(_DIR))


def _discover_models() -> list:
    models = []
    for subdir in sorted(_DIR.iterdir()):
        if not subdir.is_dir() or subdir.name.startswith("_") or subdir.name == "tools":
            continue
        model_path = subdir / "model.py"
        if not model_path.exists():
            continue
        spec = importlib.util.spec_from_file_location(subdir.name, model_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[subdir.name] = mod
        try:
            spec.loader.exec_module(mod)
        except Exception as e:
            print(f"[warn] could not import {model_path}: {e}", file=sys.stderr)
            continue
        if hasattr(mod, "OPTIMIZATION_NUMBER") and hasattr(mod, "KNOB_NAMES"):
            models.append(mod)
    models.sort(key=lambda m: m.OPTIMIZATION_NUMBER)
    return models


def build_template() -> dict:
    schema = {}
    for mod in _discover_models():
        key = f"opt{mod.OPTIMIZATION_NUMBER}"
        entry = {"applicable": None}
        for knob in mod.KNOB_NAMES:
            entry[knob] = None
        schema[key] = entry
    return schema


def build_guide() -> str:
    lines = [
        "# Knob Extraction Guide (v2)",
        "",
        "Each NKI optimization performance model in v2 has a typed `Knobs` dataclass",
        "of structural facts the agent must extract from kernel source. Profile",
        "metrics (latency, engine times, byte counts) are NOT knobs — they come from",
        "`profile.json` (the output of `nki-profile`).",
        "",
        "Workflow:",
        "1. Read your kernel source and the profiler output.",
        "2. For each opt below, decide whether the pattern applies. Set `applicable: true`",
        "   only if the pattern is present and not already optimized.",
        "3. Fill in the knob values. Use `null` for any field you cannot determine.",
        "4. Save as `knobs.json` and run:",
        "   ```bash",
        "   ./python-in-venv nki-opt-models-v2/predict_all_v2.py \\",
        "       --profile profile.json --knobs knobs.json --output predictions.json",
        "   ```",
        "5. Read `predictions.json` — pick the highest-speedup applicable opt that has",
        "   no `extrapolation_warning`, then implement it.",
        "",
        "---",
        "",
    ]
    for mod in _discover_models():
        num = mod.OPTIMIZATION_NUMBER
        name = getattr(mod, "OPTIMIZATION_NAME", f"Opt #{num}")
        required = getattr(mod, "REQUIRED_PROFILE_FIELDS", ())
        lines.append(f"## opt{num} — {name}")
        lines.append("")
        if required:
            lines.append(f"**Required profile fields:** "
                         + ", ".join(f"`{f}`" for f in required))
            lines.append("")
        for knob, doc in mod.KNOB_NAMES.items():
            lines.append(f"- `{knob}` — {doc}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--write-guide", action="store_true",
                   help="Regenerate KNOB_GUIDE.md and write it to disk.")
    args = p.parse_args()

    if args.write_guide:
        out = _DIR / "KNOB_GUIDE.md"
        out.write_text(build_guide())
        print(f"Wrote {out}", file=sys.stderr)
        return

    print(json.dumps(build_template(), indent=2))


if __name__ == "__main__":
    main()
