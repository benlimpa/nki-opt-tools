"""Scan benchmark-opt/runs/ → flat manifest table.

CLI:
    bench-manifest                # print summary
    bench-manifest --rebuild      # rewrite manifest.csv
    bench-manifest --as-csv       # dump full table to stdout
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterator

from . import common


def iter_run_dirs(runs_dir: Path | None = None) -> Iterator[Path]:
    runs_dir = runs_dir or common.RUNS_DIR
    if not runs_dir.exists():
        return
    for p in sorted(runs_dir.iterdir()):
        if p.is_dir() and (p / "metadata.json").exists():
            yield p


def scan(runs_dir: Path | None = None) -> list[dict]:
    """One row per run × measurement, plus one row per measurement-less run."""
    rows: list[dict] = []
    for run_dir in iter_run_dirs(runs_dir):
        try:
            md = common.RunMetadata.from_path(run_dir)
        except Exception as e:
            rows.append({
                "run_id": run_dir.name,
                "_error": f"failed to parse metadata.json: {e}",
            })
            continue

        base_row = {
            "run_id":           md.run_id,
            "source":           md.source,
            "kernel":           md.kernel,
            "approach":         md.approach,
            "approach_variant": md.approach_variant,
            "model":            md.model,
            "agent_features":   md.agent_features,
            "replicate":        md.replicate,
            "timestamp":        md.timestamp,
            "package_version":  md.package_version,
            "cheat":            md.cheat,
            "cheat_reason":     md.cheat_reason,
            "run_dir":          str(run_dir.relative_to(common.BENCHMARK_OPT)),
        }

        if not md.measurements:
            rows.append({**base_row,
                         "measurement_method": None,
                         "baseline_ms":  None,
                         "optimized_ms": None,
                         "speedup":      None})
            continue

        for m in md.measurements:
            rows.append({
                **base_row,
                "measurement_method":   m.method,
                "measured_at":          m.measured_at,
                "baseline_ms":          m.baseline_ms,
                "optimized_ms":         m.optimized_ms,
                "speedup":              m.speedup,
                "per_case_baseline_ms": json.dumps(m.per_case_baseline_ms) if m.per_case_baseline_ms else None,
                "per_case_optimized_ms":json.dumps(m.per_case_optimized_ms) if m.per_case_optimized_ms else None,
                "per_case_speedups":    json.dumps(m.per_case_speedups) if m.per_case_speedups else None,
            })
    return rows


def write_manifest(rows: list[dict], path: Path | None = None) -> Path:
    path = path or common.MANIFEST_PATH
    if not rows:
        # Still write a header-only manifest so downstream tools have a file.
        fieldnames = ["run_id", "source", "kernel", "approach", "approach_variant",
                      "model", "agent_features", "measurement_method", "speedup"]
    else:
        # Union of all keys across rows (preserve first-seen order).
        seen: dict[str, None] = {}
        for r in rows:
            for k in r.keys():
                seen.setdefault(k, None)
        fieldnames = list(seen.keys())

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


def _summarize(rows: list[dict]) -> None:
    if not rows:
        print(f"No runs in {common.RUNS_DIR.relative_to(common.REPO_ROOT)}/.")
        return

    # One physical-run count = unique run_ids; rows fan out over measurements.
    run_ids = {r["run_id"] for r in rows if "_error" not in r}
    errors  = [r for r in rows if "_error" in r]

    print(f"Runs in {common.RUNS_DIR.relative_to(common.REPO_ROOT)}/: "
          f"{len(run_ids)} ({len(rows)} measurement rows)")
    if errors:
        print(f"  ERRORS: {len(errors)}")
        for e in errors[:5]:
            print(f"    {e['run_id']}: {e['_error']}")

    by_af: dict[str, int] = {}
    by_method: dict[str | None, int] = {}
    cheats = 0
    for r in rows:
        if "_error" in r:
            continue
        by_af[r["agent_features"]] = by_af.get(r["agent_features"], 0) + 1
        by_method[r["measurement_method"]] = by_method.get(r["measurement_method"], 0) + 1
        if r.get("cheat"):
            cheats += 1

    print("\n  agent_features (measurement-row counts):")
    for af, n in sorted(by_af.items()):
        print(f"    {af:<28} {n}")
    print("\n  measurement_method counts:")
    for m, n in sorted(by_method.items(), key=lambda kv: (kv[0] is None, kv[0] or "")):
        label = m if m else "(no measurement)"
        print(f"    {label:<28} {n}")
    if cheats:
        print(f"\n  cheat-flagged measurement rows: {cheats}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rebuild", action="store_true",
                   help="Write manifest.csv to benchmark-opt/.")
    p.add_argument("--as-csv", action="store_true",
                   help="Dump the full table to stdout as CSV.")
    p.add_argument("--runs-dir", type=Path, default=None,
                   help="Override runs/ location (default: benchmark-opt/runs/).")
    args = p.parse_args(argv)

    rows = scan(args.runs_dir)

    if args.as_csv:
        if rows:
            seen: dict[str, None] = {}
            for r in rows:
                for k in r.keys():
                    seen.setdefault(k, None)
            w = csv.DictWriter(sys.stdout, fieldnames=list(seen.keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
        return 0

    if args.rebuild:
        path = write_manifest(rows)
        print(f"Wrote {len(rows)} rows to {path.relative_to(common.REPO_ROOT)}")
    _summarize(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
