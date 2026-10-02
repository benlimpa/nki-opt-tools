#!/usr/bin/env python3
"""
nki-opt-models-v2/predict_all_v2.py
====================================

Centralized entrypoint for v2 performance models.

Usage:
    python predict_all_v2.py --profile profile.json --knobs knobs.json [--output predictions.json]

Inputs
------
``profile.json`` — output of ``opt_profiler._benchmark_neff()`` (the
``hardware_metrics`` field of the profiler's result dict). Contains
``latency_ms``, engine active times, byte counts, etc.

``knobs.json`` — per-opt structural knobs the agent extracted from kernel
source. Shape:

    {
      "opt16": {"applicable": true, "n_tiles": 8, "has_serial_dma_compute": true},
      "opt35": {"applicable": false}
    }

Output
------
``predictions.json``:

    {
      "ranked": [
        {
          "opt": 16, "name": "Pipeline Compute and DMA (3-Stage)",
          "applicable": true, "speedup": 1.32, "t_after_us": 31.59,
          "delta_us": {"t_total_us": -10.20, "t_tensor_us": -8.28},
          "extrapolation_warning": null,
          "notes": "DMA-bound: hides 8.28us behind DMA"
        },
        ...
      ]
    }

Sorted by descending predicted speedup, with applicable entries first.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys
from typing import Any

_MODELS_DIR = pathlib.Path(__file__).parent

# Ensure _common is importable as a top-level module from each model
sys.path.insert(0, str(_MODELS_DIR))


def _load_models() -> list:
    """Import model.py from each subdirectory; return modules with the v2 interface."""
    models = []
    for subdir in sorted(_MODELS_DIR.iterdir()):
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
            print(f"[warn] Failed to import {model_path}: {e}", file=sys.stderr)
            sys.modules.pop(subdir.name, None)
            continue
        if (hasattr(mod, "OPTIMIZATION_NUMBER")
            and hasattr(mod, "predict")
            and hasattr(mod, "Knobs")):
            models.append(mod)
    models.sort(key=lambda m: m.OPTIMIZATION_NUMBER)
    return models


def _build_knobs(mod, opt_features: dict) -> Any:
    """Construct the model's typed Knobs dataclass from the JSON dict."""
    KnobsCls = mod.Knobs
    valid = {f for f in KnobsCls.__dataclass_fields__}
    kwargs = {k: v for k, v in opt_features.items()
              if k in valid and k != "applicable"}
    return KnobsCls(**kwargs)


def _flatten_hardware_metrics(profile_metrics: dict) -> dict:
    """
    v5: ``profile.json`` shipped into the workspace as the prediction anchor
    may be either the flat shape (output of ``nki-profile``, which already
    flattens ``hardware_metrics``) or the nested shape (raw profiler dict, as
    seeded by the bench harness). Normalize to flat so downstream
    ``KernelProfile.from_metrics`` finds the engine counters.

    Pre-v5, the nested shape silently produced ``t_dma_us=None`` /
    ``t_tensor_us=None``, which made every bandwidth/compute model decline
    and funnelled the agent into opt38-only — see
    benchmark-opt/reports/mg_v2_lncfix_shortfalls.md ("Cross-cutting bug").
    """
    hw = profile_metrics.get("hardware_metrics")
    if not isinstance(hw, dict):
        return profile_metrics
    flat = dict(profile_metrics)
    flat.pop("hardware_metrics", None)
    for k, v in hw.items():
        if k == "latency_ms":
            continue
        flat.setdefault(k, v)
    return flat


def run_predictions(profile_metrics: dict, knobs_dict: dict) -> dict:
    """
    Run all v2 models against the supplied profile + knobs dicts.

    ``profile_metrics`` is the raw dict from ``_benchmark_neff()`` (it will be
    converted via ``KernelProfile.from_metrics``).

    ``knobs_dict`` is keyed by ``"opt{N}"``; each sub-dict has at minimum
    ``"applicable": bool`` plus the model's knob fields.
    """
    from _common import KernelProfile

    profile_metrics = _flatten_hardware_metrics(profile_metrics)
    profile = KernelProfile.from_metrics(profile_metrics)

    models = _load_models()
    results = []

    for mod in models:
        opt_num = mod.OPTIMIZATION_NUMBER
        opt_name = getattr(mod, "OPTIMIZATION_NAME", f"Opt #{opt_num}")
        opt_key = f"opt{opt_num}"
        opt_features = knobs_dict.get(opt_key, {})
        applicable = opt_features.get("applicable")
        auto_applied_note: str | None = None

        # v3: profile-driven applicability override. If the agent left
        # `applicable` false/null AND the model exposes
        # `infer_applicable_from_profile`, give the model a chance to mark
        # itself applicable based on profile signals (e.g. spill_save_bytes>0
        # for fusion opts, packet-to-byte ratio for opt33). The hook returns
        # either None (defer to agent) or a dict of synthesized knob values
        # that are merged into opt_features.
        if not applicable and hasattr(mod, "infer_applicable_from_profile"):
            try:
                inferred = mod.infer_applicable_from_profile(profile, opt_features)
            except Exception as e:
                print(f"[warn] opt{opt_num} infer_applicable raised: {e}",
                      file=sys.stderr)
                inferred = None
            if inferred is not None:
                opt_features = {**opt_features, **inferred, "applicable": True}
                applicable = True
                auto_applied_note = (
                    "auto-applied by profile signal "
                    "(agent marked applicable=false; v3 dispatcher overrode "
                    "based on profile metrics)"
                )

        entry = {
            "opt": opt_num,
            "name": opt_name,
            "applicable": applicable,
            "speedup": None,
            "t_after_us": None,
            "delta_us": None,
            "extrapolation_warning": None,
            "notes": None,
        }

        if not applicable:
            results.append(entry)
            continue

        try:
            knobs = _build_knobs(mod, opt_features)
        except TypeError as e:
            print(f"[warn] opt{opt_num}: bad knobs ({e})", file=sys.stderr)
            entry["applicable"] = False
            results.append(entry)
            continue

        try:
            pred = mod.predict(profile, knobs)
        except Exception as e:
            print(f"[warn] opt{opt_num} predict raised: {e}", file=sys.stderr)
            pred = None

        if pred is None:
            entry["applicable"] = False
            results.append(entry)
            continue

        notes = pred.notes
        if auto_applied_note:
            notes = f"[{auto_applied_note}] {notes}" if notes else f"[{auto_applied_note}]"
        entry.update({
            "speedup": round(pred.speedup, 3),
            "t_after_us": round(pred.t_after_us, 2),
            "delta_us": {k: round(v, 2) for k, v in pred.delta_us.items()},
            "extrapolation_warning": pred.extrapolation_warning,
            "notes": notes,
        })
        results.append(entry)

    def sort_key(r):
        if r["applicable"] and r["speedup"] is not None:
            return (0, -r["speedup"])
        return (1, 0)

    results.sort(key=sort_key)
    return {"ranked": results}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all NKI v2 performance models against a profile + knobs.",
    )
    parser.add_argument("--profile", required=True, metavar="FILE",
                        help="Path to profile.json (output of opt_profiler).")
    parser.add_argument("--knobs", required=True, metavar="FILE",
                        help="Path to knobs.json (per-opt structural knobs).")
    parser.add_argument("--output", default=None, metavar="FILE",
                        help="Write predictions JSON here. If omitted, print to stdout.")
    args = parser.parse_args()

    profile_path = pathlib.Path(args.profile)
    knobs_path = pathlib.Path(args.knobs)
    if not profile_path.exists():
        print(f"[error] profile file not found: {profile_path}", file=sys.stderr)
        sys.exit(1)
    if not knobs_path.exists():
        print(f"[error] knobs file not found: {knobs_path}", file=sys.stderr)
        sys.exit(1)

    profile_metrics = json.loads(profile_path.read_text())
    knobs_dict = json.loads(knobs_path.read_text())
    predictions = run_predictions(profile_metrics, knobs_dict)
    out_text = json.dumps(predictions, indent=2)

    if args.output:
        pathlib.Path(args.output).write_text(out_text)
        print(f"Predictions written to {args.output}", file=sys.stderr)
    else:
        print(out_text)


if __name__ == "__main__":
    main()
