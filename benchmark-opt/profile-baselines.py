"""Run profile_bench3 on every baseline (kernel + oracle) pair and cache the
result under benchmark-opt/baseline-profiles/<baseline_category>/<kernel>.json.

Kernels are discovered from the canonical registry:
- the (source, kernel) set comes from `list_kernels_on_disk(source)` for each
  source in `SOURCE_PATHS`;
- the kernel/oracle paths and baseline_category come from the per-source path
  helpers (`kernel_path_for`, `oracle_path_for`, `baseline_category_for`).

The legacy per-source kernel/oracle lists that used to live here are gone — if
a kernel is added on disk it's profiled automatically (skip it via
`kernels.toml` if it shouldn't be).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from triton_nki_transpile.bench_runs import common

REPO_ROOT = common.REPO_ROOT
OUT = common.BASELINE_PROFILES_DIR


def _targets() -> list[tuple[str, str, Path, Path]]:
    """Enumerate (baseline_category, kernel, kernel_path, oracle_path) tuples
    for every source × on-disk kernel, honoring kernels.toml `skip_default`."""
    registry = common.load_kernels_registry()
    out: list[tuple[str, str, Path, Path]] = []
    for src in common.SOURCES:
        for k in common.kernels_for_source(src, registry):
            cat   = common.baseline_category_for(src)
            kpath = common.kernel_path_for(src, k)
            opath = common.oracle_path_for(src, k, must_exist=False)
            out.append((cat, k, kpath, opath))
    return out


def run(category: str, kernel: str, kpath: Path, opath: Path) -> None:
    out_dir = OUT / category
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{kernel}.json"
    # Skip only when the cached entry is a successful profile. Failed/skipped
    # entries get re-attempted so a fix to the kernel or oracle is picked up
    # without manually deleting the cache.
    if out_file.exists():
        try:
            cached = json.loads(out_file.read_text())
            pr = (cached or {}).get("profile_result") or {}
            if pr.get("success") and pr.get("latency_ms") is not None:
                print(f"  [skip] {category}/{kernel} — already profiled")
                return
        except (OSError, json.JSONDecodeError):
            pass
        print(f"  [retry] {category}/{kernel} — previous cache was {('skipped' if cached.get('skipped') else 'a failed profile')}")
    if not kpath.exists():
        out_file.write_text(json.dumps({"category": category, "kernel": kernel,
            "skipped": True, "reason": f"no baseline kernel at {kpath}"}, indent=2))
        print(f"  [skip] {category}/{kernel} — no baseline kernel")
        return
    if not opath.exists():
        out_file.write_text(json.dumps({"category": category, "kernel": kernel,
            "skipped": True, "reason": f"no oracle at {opath}"}, indent=2))
        print(f"  [skip] {category}/{kernel} — no oracle")
        return

    kernel_code = kpath.read_text()
    bench_code = opath.read_text()
    if "bench3" not in bench_code:
        out_file.write_text(json.dumps({"category": category, "kernel": kernel,
            "skipped": True, "reason": "oracle has no bench3"}, indent=2))
        print(f"  [skip] {category}/{kernel} — oracle has no bench3")
        return

    # Lazy import — Neuron deps only loaded when actually profiling.
    from triton_nki_transpile.opt_profiler import NKIOptimizationProfiler
    from triton_nki_transpile.utils import extract_wrapper_name

    try:
        wrapper_name = extract_wrapper_name(kernel_code)
    except Exception as e:
        out_file.write_text(json.dumps({"category": category, "kernel": kernel,
            "skipped": True, "reason": f"could not extract wrapper name: {e}"}, indent=2))
        print(f"  [skip] {category}/{kernel} — wrapper extraction failed: {e}")
        return
    if not wrapper_name:
        out_file.write_text(json.dumps({"category": category, "kernel": kernel,
            "skipped": True, "reason": "wrapper name not found"}, indent=2))
        return

    print(f"  [run]  {category}/{kernel}  wrapper={wrapper_name}", flush=True)
    t0 = time.time()
    profiler = NKIOptimizationProfiler()
    result = profiler.profile_bench3(
        kernel_code=kernel_code,
        bench_code=bench_code,
        wrapper_name=wrapper_name,
        timeout=600,
    )
    elapsed = time.time() - t0
    record = {
        "category": category, "kernel": kernel,
        "kernel_path": str(kpath.relative_to(REPO_ROOT)),
        "oracle_path": str(opath.relative_to(REPO_ROOT)),
        "wrapper_name": wrapper_name,
        "elapsed_sec": round(elapsed, 1),
        "profile_result": result,
    }
    out_file.write_text(json.dumps(record, indent=2))
    if result.get("success") and result.get("latency_ms") is not None:
        print(f"  [ok]   {category}/{kernel}  mean={result['latency_ms']:.3f}ms  per-case={result.get('per_case_latency_ms')}  ({elapsed:.0f}s)", flush=True)
    elif result.get("success"):
        print(f"  [ok?]  {category}/{kernel}  no latency: {result.get('error','')[:80]}", flush=True)
    else:
        err = (result.get("error") or "")[:200].replace("\n", " ")
        print(f"  [FAIL] {category}/{kernel}  {err}", flush=True)


def main() -> None:
    only = set(sys.argv[1:]) if len(sys.argv) > 1 else None
    for category, kernel, kpath, opath in _targets():
        if only and f"{category}/{kernel}" not in only and kernel not in only:
            continue
        run(category, kernel, kpath, opath)


if __name__ == "__main__":
    main()
