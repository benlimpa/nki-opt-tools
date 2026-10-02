#!/usr/bin/env python3
"""
nki-opt-models-v2/tools/run_calibration_sweep.py
================================================

Profile a sweep of (before, after) kernel pairs and append rows to a model's
``calibration.csv``.

Each sweep entry is one (kernel_id, knob values, before kernel path, after
kernel path, oracle path) tuple. The runner invokes ``nki-profile`` on each
kernel, parses its JSON output, and writes one CSV row per variant.

Usage:
    python tools/run_calibration_sweep.py \\
        --plan opt16-pipeline-compute-dma/calibration_plan.yaml \\
        [--csv opt16-pipeline-compute-dma/calibration.csv] \\
        [--dry-run]

calibration_plan.yaml shape (list of entries; YAML-or-JSON-compatible):

    # one row per pair to profile
    - kernel_id: mm_K128_M1024
      knobs: {n_tiles: 8, tile_m: 128, dtype: fp32}
      before: sweep_kernels/mm_K128_M1024_before.py
      after:  sweep_kernels/mm_K128_M1024_after.py
      oracle: sweep_kernels/mm_K128_M1024_oracle.py

Paths in the plan are resolved relative to the plan file's directory.

The CSV is appended to (created if missing). Existing rows for the same
(kernel_id, variant) pair are overwritten — re-running the sweep is idempotent.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import shlex
import subprocess
import sys

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None


# Columns written for every CSV row. Per-knob columns are added dynamically
# (one per knob name found across all entries in the plan).
_FIXED_COLUMNS = [
    "kernel_id", "variant",
    "t_total_us", "t_active_us",
    "t_tensor_us", "t_vector_us", "t_scalar_us", "t_gpsimd_us",
    "t_dma_us", "t_static_dma_us",
    "hbm_read_bytes", "hbm_write_bytes",
    "dma_packet_count", "static_dma_packet_count",
    "vector_instr_count", "tensor_instr_count", "matmul_instr_count",
]

# Map (csv_column, profile_field, conversion). Conversion is "ms_to_us",
# "sec_to_us", or "int" — matches KernelProfile.from_metrics logic.
_PROFILE_MAP: list[tuple[str, str, str]] = [
    ("t_total_us", "latency_ms", "ms_to_us"),
    ("t_active_us", "total_active_time_ms", "ms_to_us"),
    ("t_tensor_us", "tensor_engine_active_time", "sec_to_us"),
    ("t_vector_us", "vector_engine_active_time", "sec_to_us"),
    ("t_scalar_us", "scalar_engine_active_time", "sec_to_us"),
    ("t_gpsimd_us", "gpsimd_engine_active_time", "sec_to_us"),
    ("t_dma_us", "dma_active_time", "sec_to_us"),
    ("t_static_dma_us", "static_dma_active_time", "sec_to_us"),
    ("hbm_read_bytes", "hbm_read_bytes", "int"),
    ("hbm_write_bytes", "hbm_write_bytes", "int"),
    ("dma_packet_count", "dynamic_dma_packet_count", "int"),
    ("static_dma_packet_count", "static_dma_packet_count", "int"),
    ("vector_instr_count", "vector_engine_instruction_count", "int"),
    ("tensor_instr_count", "tensor_engine_instruction_count", "int"),
    ("matmul_instr_count", "matmul_instruction_count", "int"),
]


def _convert(value, kind: str):
    if value is None:
        return None
    if kind == "ms_to_us":
        return round(float(value) * 1e3, 4)
    if kind == "sec_to_us":
        return round(float(value) * 1e6, 4)
    if kind == "int":
        return int(value)
    raise ValueError(f"unknown conversion {kind}")


def _load_plan(path: pathlib.Path) -> list[dict]:
    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        if yaml is None:
            print("[error] PyYAML not installed; pass a .json plan or `pip install pyyaml`",
                  file=sys.stderr)
            sys.exit(1)
        return yaml.safe_load(text)
    return json.loads(text)


def _run_nki_profile(kernel_path: pathlib.Path, oracle_path: pathlib.Path,
                     timeout: int = 600) -> dict:
    """Invoke nki-profile and return the parsed JSON dict."""
    cmd = [
        "nki-profile",
        "--kernel-path", str(kernel_path),
        "--oracle-path", str(oracle_path),
        "--bench",
    ]
    print(f"  $ {' '.join(shlex.quote(x) for x in cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        return {"success": False, "error": proc.stderr.strip()[:300]}
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        return {"success": False, "error": f"could not parse nki-profile output: {e}"}


def _build_row(kernel_id: str, variant: str, knobs: dict,
               metrics: dict, knob_columns: list[str]) -> dict:
    row = {"kernel_id": kernel_id, "variant": variant}
    for k in knob_columns:
        row[k] = knobs.get(k)
    for col, field, kind in _PROFILE_MAP:
        row[col] = _convert(metrics.get(field), kind)
    return row


def _read_existing(csv_path: pathlib.Path) -> tuple[list[str], list[dict]]:
    if not csv_path.exists():
        return [], []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def _write_csv(csv_path: pathlib.Path, columns: list[str], rows: list[dict]) -> None:
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for r in rows:
            writer.writerow({c: r.get(c) for c in columns})


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Profile a calibration sweep and append rows to calibration.csv."
    )
    parser.add_argument("--plan", required=True, type=pathlib.Path,
                        help="Path to calibration_plan.yaml (or .json).")
    parser.add_argument("--csv", default=None, type=pathlib.Path,
                        help="Output CSV path. Defaults to plan_dir/calibration.csv.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan resolution but do not invoke nki-profile.")
    parser.add_argument("--timeout", type=int, default=600,
                        help="Per-kernel profile timeout (seconds).")
    args = parser.parse_args()

    plan_path = args.plan.resolve()
    if not plan_path.exists():
        print(f"[error] plan not found: {plan_path}", file=sys.stderr)
        return 1
    plan_dir = plan_path.parent
    csv_path = (args.csv or (plan_dir / "calibration.csv")).resolve()

    entries = _load_plan(plan_path)
    if not isinstance(entries, list) or not entries:
        print(f"[error] plan must be a non-empty list", file=sys.stderr)
        return 1

    knob_columns = sorted({k for e in entries for k in (e.get("knobs") or {})})
    columns = ["kernel_id"] + knob_columns + ["variant"] + [
        c for c in _FIXED_COLUMNS if c not in ("kernel_id", "variant")
    ]

    _, existing_rows = _read_existing(csv_path)
    by_key: dict[tuple[str, str], dict] = {
        (r["kernel_id"], r["variant"]): r for r in existing_rows
    }

    for entry in entries:
        kernel_id = entry["kernel_id"]
        knobs = entry.get("knobs") or {}
        oracle = (plan_dir / entry["oracle"]).resolve()
        for variant in ("before", "after"):
            kernel = (plan_dir / entry[variant]).resolve()
            print(f"[{kernel_id}/{variant}] knobs={knobs}")
            print(f"  kernel: {kernel}")
            print(f"  oracle: {oracle}")
            if args.dry_run:
                continue
            if not kernel.exists():
                print(f"  [skip] kernel file missing")
                continue
            if not oracle.exists():
                print(f"  [skip] oracle file missing")
                continue
            metrics = _run_nki_profile(kernel, oracle, timeout=args.timeout)
            if not metrics.get("success"):
                print(f"  [warn] profile failed: {metrics.get('error')}")
                continue
            row = _build_row(kernel_id, variant, knobs, metrics, knob_columns)
            by_key[(kernel_id, variant)] = row
            print(f"  -> t_total_us={row['t_total_us']}, t_dma_us={row['t_dma_us']}, "
                  f"t_tensor_us={row['t_tensor_us']}")

    if args.dry_run:
        print(f"\n[dry-run] would write {len(entries) * 2} rows to {csv_path}")
        return 0

    rows = sorted(by_key.values(), key=lambda r: (r["kernel_id"], r["variant"]))
    _write_csv(csv_path, columns, rows)
    print(f"\nWrote {len(rows)} rows to {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
