"""Generate the canonical results CSV from runs/ — replaces hand-editing.

Filters runs by `agent_features` and `measurement_method` (both required), then
aggregates replicates into one row per (source, kernel) with one column per
approach (or approach@variant). Output schema mirrors
`NKI Optimization Results - Codex-Speedup.csv`:

    Source, Kernel, Baseline (ms),
    NoInfo (ms), AutoComp (ms), PreExplored (ms), ModelGuided (ms),
    NoInfo Speedup, AutoComp Speedup, PreExplored Speedup, ModelGuided Speedup,
    [per-case latency columns]

CLI:
    bench-report \\
        --agent-features codex_v2_opt38 \\
        --measurement-method bench3_geomean \\
        --model gpt-5.4 \\
        --agg geomean \\
        --out reports/codex_v2_opt38__bench3_geomean__geomean.csv

    # Or emit one CSV per cell in matrix.toml:
    bench-report --from-matrix --agg geomean
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Optional

from . import common
from .common import (
    BENCHMARK_OPT, REPORTS_DIR, SOURCES, APPROACHES, APPROACH_LABELS,
    REPO_ROOT, RunMetadata,
)
from .manifest import iter_run_dirs


# Source-name display: underscores → hyphens (matches canonical CSV).
def _source_display(s: str) -> str:
    return s.replace("_", "-")


def _geomean(xs: list[float]) -> float:
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


AGGREGATIONS = {
    "first":   lambda xs: xs[0] if xs else None,
    "last":    lambda xs: xs[-1] if xs else None,
    "best":    lambda xs: max(xs) if xs else None,
    "worst":   lambda xs: min(xs) if xs else None,
    "median":  lambda xs: statistics.median(xs) if xs else None,
    "mean":    lambda xs: statistics.mean(xs) if xs else None,
    "geomean": lambda xs: _geomean(xs) if xs else None,
}


def collect(*, agent_features: str, measurement_method: str,
            model: str, include_cheats: bool) -> list[RunMetadata]:
    """Return the runs matching the filter, with the chosen measurement attached.

    A run is dropped if it doesn't have a measurement under measurement_method.
    """
    out: list[RunMetadata] = []
    for run_dir in iter_run_dirs():
        try:
            md = RunMetadata.from_path(run_dir)
        except Exception:
            continue
        if md.agent_features != agent_features:
            continue
        if md.model != model:
            continue
        if md.cheat and not include_cheats:
            continue
        if not md.get_measurement(measurement_method):
            continue
        out.append(md)
    return out


def aggregate_optimized_ms(values: list[float], agg: str) -> Optional[float]:
    """Aggregate optimized_ms values. For latency, lower is better, so:
       - "best" = lowest, "worst" = highest, regardless of agg meaning for speedup.
    """
    if not values:
        return None
    if agg == "best":
        return min(values)
    if agg == "worst":
        return max(values)
    if agg in ("median", "mean", "geomean", "first", "last"):
        return AGGREGATIONS[agg](values)
    raise ValueError(f"Unknown aggregation: {agg}")


def aggregate_speedup(values: list[float], agg: str) -> Optional[float]:
    """Aggregate speedup values. Higher is better."""
    if not values:
        return None
    if agg == "best":
        return max(values)
    if agg == "worst":
        return min(values)
    if agg == "geomean":
        return _geomean([v for v in values if v > 0])
    return AGGREGATIONS[agg](values)


def build_table(runs: list[RunMetadata], measurement_method: str, agg: str,
                approach_specs: list[str] | None = None) -> list[dict]:
    """Produce one row per (source, kernel) cell with per-approach columns.

    `approach_specs` are matrix-form strings (bare name = default variant, or
    `name@variant`). Each becomes one column group; the column key in the
    output rows is the canonical display form (same as in matrix.toml).
    """
    if approach_specs is None:
        approach_specs = APPROACHES
    resolved = common.resolve_approach_specs(approach_specs)

    # Bucket by (source, kernel, display) → list of measurement entries.
    buckets: dict[tuple[str, str], dict[str, list[dict]]] = {}
    all_kernels: dict[str, set[str]] = {}  # source -> kernels seen

    for md in runs:
        m = md.get_measurement(measurement_method)
        if not m or m.baseline_ms is None or m.optimized_ms is None:
            continue
        # Compute this run's display key. Skip if it doesn't match a requested column.
        run_display = common.format_approach_spec(md.approach, md.approach_variant)
        # Only bucket if this display is in the requested set.
        if run_display not in {d for _, _, d in resolved}:
            continue
        key = (md.source, md.kernel)
        buckets.setdefault(key, {}).setdefault(run_display, []).append({
            "baseline_ms":  m.baseline_ms,
            "optimized_ms": m.optimized_ms,
            "speedup":      m.speedup,
            "per_case_baseline_ms":  m.per_case_baseline_ms,
            "per_case_optimized_ms": m.per_case_optimized_ms,
            "per_case_speedups":     m.per_case_speedups,
        })
        all_kernels.setdefault(md.source, set()).add(md.kernel)

    rows: list[dict] = []
    # Order rows: SOURCES first, then any extras alphabetically; kernels alpha within each source.
    source_order = [s for s in SOURCES if s in all_kernels] + sorted(set(all_kernels) - set(SOURCES))
    for src in source_order:
        for kernel in sorted(all_kernels[src]):
            row: dict = {"source": src, "kernel": kernel}

            # Baseline = aggregate across all approaches + replicates of this (source, kernel).
            baselines: list[float] = []
            for entries in buckets[(src, kernel)].values():
                baselines.extend(e["baseline_ms"] for e in entries if e["baseline_ms"])
            row["baseline_ms"] = (statistics.median(baselines) if baselines else None)

            for _, _, display in resolved:
                entries = buckets[(src, kernel)].get(display, [])
                opt_values = [e["optimized_ms"] for e in entries if e["optimized_ms"]]
                spd_values = [e["speedup"] for e in entries if e["speedup"]]
                row[f"{display}_ms"] = aggregate_optimized_ms(opt_values, agg)
                row[f"{display}_speedup"] = aggregate_speedup(spd_values, agg)
                row[f"{display}_n"] = len(entries)

                # Per-case: pick the replicate matching the agg if all entries
                # carry per-case lists (use the same selection criterion as agg).
                pc_opt = [e["per_case_optimized_ms"] for e in entries if e["per_case_optimized_ms"]]
                pc_spd = [e["per_case_speedups"] for e in entries if e["per_case_speedups"]]
                row[f"{display}_per_case_ms"]       = _pick_per_case(pc_opt, opt_values, agg, lower_better=True)
                row[f"{display}_per_case_speedups"] = _pick_per_case(pc_spd, spd_values, agg, lower_better=False)
            rows.append(row)
    return rows


def _column_label(display: str) -> str:
    """Render a display key like "model_guided_v2@v2" to a header label
    like "ModelGuidedV2@v2". Falls back to display verbatim if unknown."""
    name, variant = common.parse_approach_spec(display)
    base = APPROACH_LABELS.get(name, name)
    return base if variant is None else f"{base}@{variant}"


def _pick_per_case(per_case_lists: list[list[float]],
                   scalar_values: list[float],
                   agg: str,
                   lower_better: bool) -> Optional[list[float]]:
    """Pick the per-case list whose scalar matches the chosen aggregation."""
    if not per_case_lists or not scalar_values:
        return None
    if agg == "first":
        return per_case_lists[0]
    if agg == "last":
        return per_case_lists[-1]
    if agg == "best":
        target = min(scalar_values) if lower_better else max(scalar_values)
    elif agg == "worst":
        target = max(scalar_values) if lower_better else min(scalar_values)
    else:
        # median/mean/geomean — synthesize a per-case row by elementwise agg.
        if not all(len(pc) == len(per_case_lists[0]) for pc in per_case_lists):
            return None
        n = len(per_case_lists[0])
        result = []
        for i in range(n):
            col = [pc[i] for pc in per_case_lists]
            if agg == "median":
                result.append(statistics.median(col))
            elif agg == "mean":
                result.append(statistics.mean(col))
            elif agg == "geomean":
                result.append(_geomean([v for v in col if v > 0]))
        return result
    # Find the per_case_list whose scalar equals target.
    for pc, sv in zip(per_case_lists, scalar_values):
        if sv == target:
            return pc
    return per_case_lists[0]


# ---------- CSV writing ----------

def write_csv(rows: list[dict], path: Path, *, agent_features: str,
              measurement_method: str, model: str, agg: str,
              approach_specs: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)

        # Header / metadata block
        w.writerow([f"# NKI Optimization Results — generated by bench_runs.report"])
        w.writerow([f"# agent_features={agent_features}"])
        w.writerow([f"# measurement_method={measurement_method}"])
        w.writerow([f"# model={model}"])
        w.writerow([f"# aggregation={agg}"])
        w.writerow([])

        # Latency + speedup summary table — the same shape as the hand-edited
        # canonical CSV. Each "column" is one approach spec (bare name or
        # name@variant) as it appears in matrix.toml.
        latency_approaches = approach_specs if approach_specs is not None \
            else ["no_info", "autocomp", "pre_explored", "model_guided"]
        # Compute the display key per spec (matches build_table's bucket key).
        displays = [d for _, _, d in common.resolve_approach_specs(latency_approaches)]

        header_top = [""] * 2 + ["Latency"] + [""] * (len(displays) + 1) + ["SpeedUp"]
        w.writerow(header_top)
        header = ["Source", "Kernel", "Baseline (ms)"]
        header.extend(f"{_column_label(d)} (ms)" for d in displays)
        header.append("")  # spacer
        header.extend(f"{_column_label(d)} Speedup" for d in displays)
        header.append("")
        header.extend(f"n {_column_label(d)}" for d in displays)
        w.writerow(header)

        for row in rows:
            cells: list = [_source_display(row["source"]), row["kernel"], _fmt_ms(row["baseline_ms"])]
            cells.extend(_fmt_ms(row[f"{d}_ms"]) for d in displays)
            cells.append("")
            cells.extend(_fmt_speedup(row[f"{d}_speedup"]) for d in displays)
            cells.append("")
            cells.extend(row[f"{d}_n"] for d in displays)
            w.writerow(cells)

        # Geomean line — geomean of per-approach speedups across all kernels.
        w.writerow([])
        w.writerow(["", "Geomean across kernels", ""])
        all_speedups: dict[str, list[float]] = {d: [] for d in displays}
        for row in rows:
            for d in displays:
                v = row[f"{d}_speedup"]
                if v and v > 0:
                    all_speedups[d].append(v)
        gm_row = ["", "geomean speedup", ""] + [""] * len(displays) + [""]
        gm_row.extend(_fmt_speedup(_geomean(all_speedups[d])) if all_speedups[d] else "—" for d in displays)
        w.writerow(gm_row)

        # Per-case latency block (matches the canonical CSV's tail section).
        w.writerow([])
        w.writerow(["Per-Case Latencies (ms)"])
        w.writerow(["Source", "Kernel"]
                   + [f"{_column_label(d)} per-case (ms)" for d in displays])
        for row in rows:
            cells = [_source_display(row["source"]), row["kernel"]]
            for d in displays:
                pc = row[f"{d}_per_case_ms"]
                cells.append(_fmt_per_case(pc))
            w.writerow(cells)

        # Per-case speedups
        w.writerow([])
        w.writerow(["Per-Case Speedups"])
        w.writerow(["Source", "Kernel"]
                   + [f"{_column_label(d)} per-case speedups" for d in displays])
        for row in rows:
            cells = [_source_display(row["source"]), row["kernel"]]
            for d in displays:
                pc = row[f"{d}_per_case_speedups"]
                cells.append(_fmt_per_case(pc))
            w.writerow(cells)


def _fmt_ms(v: Optional[float]) -> str:
    return f"{v:.3f}" if isinstance(v, (int, float)) else "—"

def _fmt_speedup(v: Optional[float]) -> str:
    return f"{v:.2f}" if isinstance(v, (int, float)) else "—"

def _fmt_per_case(v: Optional[list[float]]) -> str:
    if not v:
        return ""
    return "[" + ",".join(f"{x:.3f}" for x in v) + "]"


# ---------- CLI ----------

def _generate_one(*, agent_features: str, measurement_method: str, model: str,
                  agg: str, include_cheats: bool, out: Path,
                  approach_specs: list[str] | None = None) -> tuple[int, int]:
    """Generate one CSV. Returns (n_rows, n_runs)."""
    common.require_agent_features(agent_features)
    common.require_measurement_method(measurement_method)
    runs = collect(agent_features=agent_features,
                   measurement_method=measurement_method,
                   model=model,
                   include_cheats=include_cheats)
    if not runs:
        return 0, 0
    rows = build_table(runs, measurement_method, agg, approach_specs=approach_specs)
    write_csv(rows, out,
              agent_features=agent_features,
              measurement_method=measurement_method,
              model=model,
              agg=agg,
              approach_specs=approach_specs)
    return len(rows), len(runs)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--from-matrix", action="store_true",
                   help="emit one CSV per cell in matrix.toml (ignores --agent-features/--measurement-method/--model/--out)")
    p.add_argument("--agent-features", help="agent_features tag (required unless --from-matrix)")
    p.add_argument("--measurement-method", help="measurement_method tag (required unless --from-matrix)")
    p.add_argument("--model", help="exact model string, e.g. gpt-5.4 (required unless --from-matrix)")
    p.add_argument("--agg", default="last", choices=list(AGGREGATIONS.keys()),
                   help="how to aggregate replicates (default: last)")
    p.add_argument("--include-cheats", action="store_true",
                   help="include runs flagged as cheat (excluded by default)")
    p.add_argument("--out", type=Path, default=None,
                   help="output CSV path (default: reports/<agent_features>__<method>__<agg>.csv). Ignored under --from-matrix.")
    args = p.parse_args(argv)

    if args.from_matrix:
        cells, _ = common.load_matrix_cells()
        if not cells:
            print("matrix.toml has no [[cell]] entries — nothing to report.")
            return 1
        total_rows = 0
        wrote = 0
        for cell in cells:
            name = cell["name"]
            af = cell["agent_features"]
            method = cell["measurement_method"]
            model = cell["model"]
            out = REPORTS_DIR / f"{name}__{args.agg}.csv"
            n_rows, n_runs = _generate_one(
                agent_features=af, measurement_method=method, model=model,
                agg=args.agg, include_cheats=args.include_cheats, out=out,
                approach_specs=cell.get("approaches"),
            )
            if n_runs == 0:
                print(f"[{name}] {af} / {method} / {model}: no matching runs — skipped")
                continue
            print(f"[{name}] {af} / {method} / {model}: {n_rows} rows ({n_runs} runs) -> {out.relative_to(REPO_ROOT)}")
            total_rows += n_rows
            wrote += 1
        print(f"\nWrote {wrote}/{len(cells)} CSVs ({total_rows} rows total).")
        return 0 if wrote else 1

    missing = [name for name, val in [("--agent-features", args.agent_features),
                                       ("--measurement-method", args.measurement_method),
                                       ("--model", args.model)] if not val]
    if missing:
        p.error(f"the following arguments are required (unless --from-matrix): {', '.join(missing)}")

    out = args.out or (REPORTS_DIR / f"{args.agent_features}__{args.measurement_method}__{args.agg}.csv")
    n_rows, n_runs = _generate_one(
        agent_features=args.agent_features,
        measurement_method=args.measurement_method,
        model=args.model,
        agg=args.agg,
        include_cheats=args.include_cheats,
        out=out,
    )
    if n_runs == 0:
        print(f"No runs match the filter "
              f"(agent_features={args.agent_features}, "
              f"measurement_method={args.measurement_method}, "
              f"model={args.model}, "
              f"include_cheats={args.include_cheats}).")
        return 1
    print(f"Wrote {n_rows} rows ({n_runs} matched runs) to {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
