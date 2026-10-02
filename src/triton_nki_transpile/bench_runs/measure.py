"""Append a new measurement record to existing runs by re-profiling saved kernels.

This is the cheap, retroactive path for new measurement methodologies — no
agent quota burned. The agent's saved optimized.py + history.json are unchanged;
we just run a different profiler against them and append the result to
metadata.json's `measurements` list.

CLI:
    # Add bench3_geomean to every run that doesn't have one yet
    bench-measure --method bench3_geomean --skip-if-present

    # Force-overwrite an existing measurement (e.g. profiler bug fix)
    bench-measure --method bench3_geomean --force

    # Filter by agent_features / model / source
    bench-measure --method bench3_geomean \\
        --agent-features codex_v1_opt37 --model gpt-5.4

    # Recompute speedup from saved per-case latencies (no profiling)
    bench-measure --method bench3_geomean --recompute-only
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from . import common
from .common import (
    BENCHMARK_OPT, REPO_ROOT,
    Measurement, RunMetadata,
    oracle_path_for,
)
from .manifest import iter_run_dirs


def _measurement_from_history(history: list, method: str, profiler_version: str | None = None) -> Optional[Measurement]:
    """Extract baseline + best from the history's (already-populated) profile_result fields."""
    from math import log, exp

    if not history:
        return None
    base = next((h for h in history if h.get("iteration") == 0), None)
    if not base or not (base.get("profile_result") or {}).get("success"):
        return None
    base_pr = base["profile_result"]

    best_pr = base_pr
    best_mean = base_pr.get("latency_ms")
    for entry in history:
        if entry.get("iteration") == 0 or not entry.get("accepted"):
            continue
        pr = entry.get("profile_result") or {}
        if not pr.get("success"):
            continue
        m = pr.get("latency_ms")
        if m is None:
            continue
        if best_mean is None or m < best_mean:
            best_mean = m
            best_pr = pr

    base_ms = base_pr.get("latency_ms")
    best_ms = best_pr.get("latency_ms")
    base_pc = base_pr.get("per_case_latency_ms")
    best_pc = best_pr.get("per_case_latency_ms")

    speedup, pc_speedups = None, None
    if base_pc and best_pc and len(base_pc) == len(best_pc) and all(v > 0 for v in best_pc):
        ratios = [b / c for b, c in zip(base_pc, best_pc)]
        speedup = round(exp(sum(log(r) for r in ratios) / len(ratios)), 3)
        pc_speedups = [round(r, 3) for r in ratios]
    elif base_ms and best_ms and best_ms > 0:
        speedup = round(base_ms / best_ms, 3)

    return Measurement(
        method=method,
        measured_at=datetime.now(tz=timezone.utc).isoformat(),
        baseline_ms=base_ms,
        optimized_ms=best_ms,
        speedup=speedup,
        per_case_baseline_ms=list(base_pc) if base_pc else None,
        per_case_optimized_ms=list(best_pc) if best_pc else None,
        per_case_speedups=pc_speedups,
        profiler_version=profiler_version,
    )


def _profile_history_with_method(history: list, method: str,
                                  source: str, kernel: str,
                                  timeout: int) -> tuple[list, str]:
    """Re-profile every iteration in `history` using the chosen method.

    Returns (updated_history, profiler_version). Modifies entries' profile_result in place.
    """
    method_cfg = common.require_measurement_method(method)
    profiling = method_cfg.get("profiling")

    oracle = oracle_path_for(source, kernel)  # raises if missing
    test_code = oracle.read_text()
    if profiling == "bench3" and "bench3" not in test_code:
        raise RuntimeError(f"Oracle {oracle} does not define bench3() — cannot profile under bench3 method.")

    # Lazy import — Neuron deps only loaded when actually profiling.
    from triton_nki_transpile.opt_profiler import NKIOptimizationProfiler  # noqa: PLC0415
    from triton_nki_transpile.utils import extract_wrapper_name             # noqa: PLC0415

    profiler = NKIOptimizationProfiler()
    profiler_version = "bench3-v1" if profiling == "bench3" else "single-shape-v1"

    baseline = next((h for h in history if h.get("iteration") == 0), None)
    if not baseline or not baseline.get("code"):
        raise RuntimeError("history.json has no baseline (iteration=0) with code")
    wrapper_name = extract_wrapper_name(baseline["code"])
    if not wrapper_name:
        raise RuntimeError("could not determine wrapper_name from baseline code")

    for entry in history:
        code = entry.get("code")
        if not code:
            continue
        if profiling == "bench3":
            result = profiler.profile_bench3(
                kernel_code=code, bench_code=test_code,
                wrapper_name=wrapper_name, timeout=timeout,
            )
        else:
            result = profiler.profile(
                kernel_code=code, bench_code=test_code,
                wrapper_name=wrapper_name, timeout=timeout,
            )
        entry["profile_result"] = result

    return history, profiler_version


def measure_run(run_dir: Path, *, method: str,
                timeout: int, force: bool, recompute_only: bool,
                skip_if_present: bool) -> str:
    """Apply a single run's re-measurement. Returns one of: 'added', 'skipped', 'recomputed', 'failed'."""
    md = RunMetadata.from_path(run_dir)

    existing = md.get_measurement(method)
    if existing and skip_if_present:
        return "skipped"
    if existing and not force and not recompute_only:
        return "skipped"

    history_path = run_dir / "history.json"
    if not history_path.exists():
        return "failed"
    history = json.loads(history_path.read_text())

    if recompute_only:
        # Just rebuild the measurement summary from existing profile_result fields.
        new_m = _measurement_from_history(history, method)
        if new_m is None:
            return "failed"
        md.upsert_measurement(new_m)
        (run_dir / "metadata.json").write_text(md.to_json())
        return "recomputed"

    # Re-profile every iteration with the chosen method, persist the updated
    # per-iteration profile_result to a side-file so we don't clobber the
    # original history (which is the immutable agent record).
    history_for_method, profiler_version = _profile_history_with_method(
        history=[dict(e) for e in history],  # deep-ish copy
        method=method, source=md.source, kernel=md.kernel,
        timeout=timeout,
    )
    side_file = f"history.{method}.json"
    (run_dir / side_file).write_text(json.dumps(history_for_method, indent=2))

    new_m = _measurement_from_history(history_for_method, method, profiler_version=profiler_version)
    if new_m is None:
        return "failed"
    new_m.profile_result_file = side_file
    md.upsert_measurement(new_m)
    (run_dir / "metadata.json").write_text(md.to_json())
    return "added"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", required=True, help="measurement_method tag to apply/refresh")
    p.add_argument("--agent-features", default=None, help="filter to runs with this agent_features tag")
    p.add_argument("--model", default=None, help="filter to runs with this model")
    p.add_argument("--source", default=None, help="filter to runs from this source")
    p.add_argument("--kernel", default=None, help="filter to runs of this kernel")
    p.add_argument("--approach", default=None, help="filter to runs of this approach")
    p.add_argument("--approach-variant", default=None,
                   help="filter to runs of this approach variant (e.g. v2)")
    p.add_argument("--include-cheats", action="store_true", help="include cheat-flagged runs")
    p.add_argument("--timeout", type=int, default=600, help="per-iteration profile timeout (s)")
    p.add_argument("--force", action="store_true", help="overwrite existing measurement under this method")
    p.add_argument("--skip-if-present", action="store_true",
                   help="leave runs alone if they already have a measurement under this method (default behavior)")
    p.add_argument("--recompute-only", action="store_true",
                   help="don't re-profile; just rebuild the measurement summary from existing history fields")
    p.add_argument("--limit", type=int, default=None, help="process at most N runs (debug)")
    args = p.parse_args(argv)

    common.require_measurement_method(args.method)

    targets: list[Path] = []
    for run_dir in iter_run_dirs():
        try:
            md = RunMetadata.from_path(run_dir)
        except Exception:
            continue
        if md.cheat and not args.include_cheats:
            continue
        if args.agent_features and md.agent_features != args.agent_features:
            continue
        if args.model  and md.model  != args.model:  continue
        if args.source and md.source != args.source: continue
        if args.kernel   and md.kernel           != args.kernel:           continue
        if args.approach and md.approach         != args.approach:         continue
        if args.approach_variant and md.approach_variant != args.approach_variant: continue
        targets.append(run_dir)
    if args.limit:
        targets = targets[:args.limit]

    print(f"Targeting {len(targets)} runs.")
    counts = {"added": 0, "skipped": 0, "recomputed": 0, "failed": 0}
    for i, run_dir in enumerate(targets, 1):
        try:
            result = measure_run(
                run_dir, method=args.method,
                timeout=args.timeout, force=args.force,
                recompute_only=args.recompute_only,
                skip_if_present=args.skip_if_present and not args.force,
            )
        except Exception as e:
            result = "failed"
            print(f"  [{i}/{len(targets)}] {run_dir.name}: FAILED ({e})")
            counts["failed"] += 1
            continue
        counts[result] = counts.get(result, 0) + 1
        if result != "skipped":
            print(f"  [{i}/{len(targets)}] {run_dir.name}: {result}")

    print(f"\nDone. {counts}")
    return 0 if counts["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
