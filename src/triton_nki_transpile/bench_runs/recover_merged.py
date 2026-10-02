"""Recover bench3 measurements from legacy `agent_bench_docker_*/results/merged/` dirs.

The original migration walked timestamp dirs and (correctly) ignored merged/.
But reprofile-results-bench3.py historically operated on whichever results
subdir had the latest mtime, which for many legacy_dirs was merged/ — so the
post-reprofile bench3 profile_result fields ended up in merged/<kernel>/<menu>_*
history.json files, NOT in the timestamp dirs.

This script scans each merged/<kernel>/<menu>_optimized.history.json, finds
the migrated run that produced it (by matching the baseline iteration's `code`
field), and appends a bench3_geomean measurement record to that run's
metadata.json — without touching its existing single_shape_meanratio record.

Idempotent: skips runs that already have a bench3_geomean measurement.

CLI:
    python -m triton_nki_transpile.bench_runs.recover_merged --dry-run
    python -m triton_nki_transpile.bench_runs.recover_merged --apply
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import common
from .common import BENCHMARK_OPT, REPO_ROOT, RunMetadata
from .manifest import iter_run_dirs
from .migrate import _measurement_from_history, _profile_kind


def _baseline_code_hash(history: list) -> str | None:
    base = next((h for h in history if h.get("iteration") == 0), None)
    code = (base or {}).get("code") or ""
    if not code:
        return None
    return hashlib.sha256(code.encode("utf-8")).hexdigest()[:16]


def _index_runs_by_baseline_hash() -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    for d in iter_run_dirs():
        hist_path = d / "history.json"
        if not hist_path.exists():
            continue
        try:
            history = json.loads(hist_path.read_text())
        except Exception:
            continue
        h = _baseline_code_hash(history)
        if h is None:
            continue
        out.setdefault(h, []).append(d)
    return out


def _iter_merged_history_files() -> list[tuple[Path, str, str]]:
    """Yield (history_path, kernel, menu) for every merged/<kernel>/<menu>_optimized.history.json."""
    out: list[tuple[Path, str, str]] = []
    for legacy_dir in sorted(BENCHMARK_OPT.iterdir()):
        if not (legacy_dir.is_dir() and legacy_dir.name.startswith("agent_bench_docker_version_")):
            continue
        merged = legacy_dir / "results" / "merged"
        if not merged.exists():
            continue
        for kdir in sorted(merged.iterdir()):
            if not kdir.is_dir():
                continue
            kernel = kdir.name
            for hist in sorted(kdir.glob("*_optimized.history.json")):
                menu = hist.name[: -len("_optimized.history.json")]
                out.append((hist, kernel, menu))
    return out


def _recover_analysis_caches(by_hash: dict[str, list[Path]], dry_run: bool) -> dict[str, int]:
    """Copy `<menu>_analysis.json`, `<menu>_kernel_comparison.md`, `<menu>_info_score.json`,
    and `<menu>_pattern_features.json` from merged/<kernel>/ to the matching run dir
    (renamed to drop the `<menu>_` prefix). LLM caches are expensive to regenerate.
    """
    import shutil
    counts = {"copied": 0, "skipped_present": 0, "no_match": 0}
    for legacy_dir in sorted(BENCHMARK_OPT.iterdir()):
        if not (legacy_dir.is_dir() and legacy_dir.name.startswith("agent_bench_docker_version_")):
            continue
        merged = legacy_dir / "results" / "merged"
        if not merged.exists():
            continue
        for kdir in sorted(merged.iterdir()):
            if not kdir.is_dir():
                continue
            kernel = kdir.name
            hist_path = next(iter(kdir.glob("*_optimized.history.json")), None)
            if hist_path is None:
                continue
            try:
                history = json.loads(hist_path.read_text())
            except Exception:
                continue
            h = _baseline_code_hash(history)
            if not h or h not in by_hash:
                continue
            for cache_path in kdir.iterdir():
                if not cache_path.is_file():
                    continue
                # Filenames look like `<menu>_analysis.json`, etc.
                for suffix in ("_analysis.json", "_kernel_comparison.md",
                               "_info_score.json", "_pattern_features.json"):
                    if cache_path.name.endswith(suffix):
                        menu = cache_path.name[: -len(suffix)]
                        # Find the run with this kernel+approach among matched candidates.
                        # Legacy `menu` value maps directly to the new `approach` field.
                        targets = [c for c in by_hash[h]
                                   if RunMetadata.from_path(c).kernel == kernel
                                   and RunMetadata.from_path(c).approach == menu]
                        if not targets:
                            counts["no_match"] += 1
                            break
                        targets.sort(key=lambda p: RunMetadata.from_path(p).replicate)
                        target = targets[0]
                        # In the new layout, drop the `<menu>_` prefix.
                        dst_name = cache_path.name[len(menu) + 1:]
                        dst = target / dst_name
                        if dst.exists():
                            counts["skipped_present"] += 1
                            break
                        if dry_run:
                            print(f"  cache + {target.name} <- {cache_path.relative_to(BENCHMARK_OPT)}")
                        else:
                            shutil.copy2(cache_path, dst)
                        counts["copied"] += 1
                        break
    return counts


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--dry-run", action="store_true")
    grp.add_argument("--apply", action="store_true")
    p.add_argument("--force", action="store_true",
                   help="overwrite existing bench3_geomean measurement on a run")
    p.add_argument("--caches-only", action="store_true",
                   help="only copy LLM analysis caches, don't add bench3 measurements")
    args = p.parse_args(argv)

    by_hash = _index_runs_by_baseline_hash()
    print(f"Indexed {sum(len(v) for v in by_hash.values())} runs in {len(by_hash)} unique baselines.")

    if args.caches_only:
        cache_counts = _recover_analysis_caches(by_hash, dry_run=args.dry_run)
        print(f"\nCache recovery: {cache_counts}")
        return 0

    counts = {"matched_added": 0, "matched_skipped_present": 0, "no_bench3_in_merged": 0,
              "no_match": 0, "ambiguous_match_resolved": 0, "ambiguous_match_unresolved": 0}

    for hist_path, kernel, menu in _iter_merged_history_files():
        try:
            history = json.loads(hist_path.read_text())
        except Exception:
            continue
        base = next((h for h in history if h.get("iteration") == 0), None)
        if not base:
            continue
        kind = _profile_kind((base.get("profile_result") or {}))
        if kind != "bench3":
            counts["no_bench3_in_merged"] += 1
            continue
        h = _baseline_code_hash(history)
        if not h:
            continue
        candidates = by_hash.get(h, [])
        # Filter to candidates whose metadata matches kernel+menu.
        matching: list[Path] = []
        for c in candidates:
            try:
                md = RunMetadata.from_path(c)
            except Exception:
                continue
            if md.kernel == kernel and md.approach == menu:
                matching.append(c)

        if not matching:
            counts["no_match"] += 1
            continue
        if len(matching) > 1:
            # Multiple replicates share the same baseline code. Pick the lowest
            # replicate index — the merged/ data was almost certainly aggregated
            # before later replicates existed.
            try:
                matching.sort(key=lambda p: RunMetadata.from_path(p).replicate)
            except Exception:
                pass
            counts["ambiguous_match_resolved"] += 1
        target = matching[0]

        md = RunMetadata.from_path(target)
        if not args.force and md.get_measurement("bench3_geomean"):
            counts["matched_skipped_present"] += 1
            continue

        new_m = _measurement_from_history(
            hist_path, "bench3_geomean",
            datetime.fromtimestamp(hist_path.stat().st_mtime, tz=timezone.utc).isoformat(),
        )
        if new_m is None:
            counts["no_match"] += 1
            continue

        if args.dry_run:
            print(f"  + {target.name} <- {hist_path.relative_to(BENCHMARK_OPT)}  (speedup={new_m.speedup})")
        else:
            md.upsert_measurement(new_m)
            (target / "metadata.json").write_text(md.to_json())
            print(f"  + {target.name} <- {hist_path.relative_to(BENCHMARK_OPT)}  (speedup={new_m.speedup})")
        counts["matched_added"] += 1

    print(f"\nResults: {counts}")

    # Also recover LLM analysis caches.
    cache_counts = _recover_analysis_caches(by_hash, dry_run=args.dry_run)
    print(f"Cache recovery: {cache_counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
