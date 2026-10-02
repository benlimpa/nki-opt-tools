"""Coverage report: matrix.toml (desired) vs runs/ manifest (actual).

For each cell in matrix.toml, classify each (source, kernel, approach[@variant])
combination as:
  - filled                : ≥ target_runs replicates AND each is measured
                            under the cell's measurement_method
  - needs more agent runs : fewer than target_runs replicates exist (expensive)
  - needs measurement     : enough replicates exist but lack the target measurement
                            (cheap — run measure.py)
  - missing entirely      : zero replicates
  - cheat-blocked         : all candidate replicates are flagged as cheat

CLI:
    bench-status
    bench-status --cell <name>     # restrict to one matrix cell by its `name`
    bench-status --runqueue-out queue.txt
                                                  # write missing cells to an
                                                  # editable queue file
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

from . import common
from .common import RunMetadata
from .manifest import iter_run_dirs


def _gather_runs() -> list[RunMetadata]:
    out: list[RunMetadata] = []
    for d in iter_run_dirs():
        try:
            out.append(RunMetadata.from_path(d))
        except Exception:
            pass
    return out


def _classify_cell(cell, idx, kernels_registry, default_target):
    """Return classification lists for one matrix cell:
    (filled, need_runs, need_measure, missing, cheat_only, no_baseline).

    `no_baseline` collects (src, kernel, approach) entries that would otherwise
    fall into `missing` or `need_runs` but whose cached baseline profile is
    absent. Those are blocked from the queue because the runner aborts when
    benchmark-opt/baseline-profiles/<category>/<kernel>.json is missing.
    Populate baselines first via `benchmark-opt/profile-baselines.py`.

    The `idx` keys runs by (model, agent_features, source, kernel, approach,
    approach_variant). matrix cells reference approaches as `name@variant`
    (no defaults exist).
    """
    af     = cell["agent_features"]
    method = cell["measurement_method"]
    model  = cell["model"]
    srcs   = cell.get("sources", common.SOURCES)
    approach_specs = cell.get("approaches", common.APPROACHES)
    target = cell.get("target_runs", default_target)

    approaches = common.resolve_approach_specs(approach_specs)
    kernels_by_src = common.resolve_cell_kernels(
        cell.get("kernels", "all"), srcs, kernels_registry,
    )

    filled, need_runs, need_measure, missing, cheat_only, no_baseline = \
        [], [], [], [], [], []
    for src in srcs:
        kernels = kernels_by_src[src]
        for kernel in kernels:
            # The baseline is shared across every approach for a (src, kernel) —
            # check once per kernel and reuse. "Usable" means file exists AND
            # records a successful profile; a failed-baseline .json is the same
            # as no file for gating purposes (the runner can't compute speedup
            # against a failure).
            baseline_exists = common.baseline_profile_is_usable(src, kernel)
            for approach_name, variant, display in approaches:
                cands = idx.get((model, af, src, kernel, approach_name, variant), [])
                non_cheat = [r for r in cands if not r.cheat]
                if cands and not non_cheat:
                    cheat_only.append((src, kernel, display, len(cands)))
                    continue
                n_runs = len(non_cheat)
                n_measured = sum(1 for r in non_cheat if r.get_measurement(method))
                if n_runs == 0:
                    if baseline_exists:
                        missing.append((src, kernel, display))
                    else:
                        no_baseline.append((src, kernel, display, 0, target))
                elif n_runs < target:
                    if baseline_exists:
                        need_runs.append((src, kernel, display, n_runs, target))
                    else:
                        no_baseline.append((src, kernel, display, n_runs, target))
                elif n_measured < target:
                    need_measure.append((src, kernel, display, n_runs, n_measured, target))
                else:
                    filled.append((src, kernel, display, n_runs))
    return filled, need_runs, need_measure, missing, cheat_only, no_baseline


def report(cell_name: str | None = None,
           runqueue_out: Path | None = None) -> int:
    cells, defaults = common.load_matrix_cells()
    default_target = defaults.get("default_target_runs", 1)
    default_reasoning = defaults.get("default_reasoning_effort", "medium")
    runs = _gather_runs()
    # Load + validate kernels.toml once; resolve_cell_kernels uses disk listings
    # (not the historical runs set) so newly added kernels appear automatically.
    kernels_registry = common.load_kernels_registry()

    idx: dict[tuple, list[RunMetadata]] = defaultdict(list)
    for r in runs:
        key = (r.model, r.agent_features, r.source, r.kernel, r.approach, r.approach_variant)
        idx[key].append(r)

    if not cells:
        print("matrix.toml has no [[cell]] entries — nothing to report.")
        return 0

    if cell_name is not None:
        known = [c["name"] for c in cells]
        if cell_name not in known:
            print(f"No matrix.toml cell named {cell_name!r}. Known: {known}",
                  file=sys.stderr)
            return 2

    overall_failed = 0
    queue_lines: list[str] = []

    for cell in cells:
        name = cell["name"]
        if cell_name is not None and name != cell_name:
            continue
        af, method, model = cell["agent_features"], cell["measurement_method"], cell["model"]
        target = cell.get("target_runs", default_target)
        # Agent (codex vs claude) is canonical to the agent_features tag, not a
        # cell field — look it up so the queue can launch the right agent.
        agent = common.require_agent_features(af)["agent"]
        reasoning = cell.get("reasoning_effort", default_reasoning)
        filled, need_runs, need_measure, missing, cheat_only, no_baseline = \
            _classify_cell(cell, idx, kernels_registry, default_target)

        print(f"\n[{name}] agent_features={af}  agent={agent}  "
              f"measurement_method={method}  model={model}  "
              f"reasoning_effort={reasoning}  target_runs={target}")

        def _fmt(group, label):
            print(f"  {label:<32} {len(group)}")
            for item in group[:10]:
                print(f"      {item}")
            if len(group) > 10:
                print(f"      ...and {len(group) - 10} more")

        _fmt(filled,        "filled")
        _fmt(need_runs,     "needs more agent runs")
        _fmt(need_measure,  "needs measurement")
        _fmt(missing,       "missing entirely")
        if cheat_only:
            _fmt(cheat_only, "cheat-only (excluded)")
        if no_baseline:
            _fmt(no_baseline, "BLOCKED: missing baseline")
            # Distinct fix hint — the user has to populate baselines, not run agents.
            print("      → fix: `python benchmark-opt/profile-baselines.py` "
                  "(only those above will be (re)profiled)")

        # Anything that would block a clean fill of the matrix counts as failure.
        # no_baseline is included so CI/automation surfaces it; it's a precondition,
        # not an agent shortfall.
        overall_failed += (len(missing) + len(need_runs) + len(need_measure)
                           + len(no_baseline))

        if runqueue_out is not None:
            queue_lines.extend(_render_queue_section(
                cell_name=name, af=af, method=method, model=model,
                agent=agent, reasoning=reasoning,
                missing=missing, need_runs=need_runs,
                no_baseline=no_baseline,
            ))

    if runqueue_out is not None:
        _write_queue(runqueue_out, queue_lines)
        print(f"\nQueue written to {runqueue_out}")

    return 0 if overall_failed == 0 else 1


def _render_queue_section(*, cell_name: str, af: str, method: str, model: str,
                          agent: str, reasoning: str,
                          missing: list, need_runs: list,
                          no_baseline: list) -> list[str]:
    """Emit a queue-file section for one matrix cell. Returns line list (no
    trailing newlines). Entries lacking a baseline profile are listed as
    commented-out lines with a fix hint — the runner aborts on missing
    baselines, so we don't emit them as runnable."""
    lines: list[str] = []
    lines.append("")
    lines.append(f"# ----- cell {cell_name} -----")
    lines.append(f"[{cell_name} / {af} / {method} / {model} / {agent} / {reasoning}]")
    if not (missing or need_runs or no_baseline):
        lines.append("# (cell fully filled — nothing to run)")
        return lines
    lines.append("# columns: source  kernel  approach[@variant]")
    # "missing entirely" first (most urgent), then "needs more runs" (partial).
    for src, kernel, approach in missing:
        lines.append(f"{src}\t{kernel}\t{approach}")
    for entry in need_runs:
        src, kernel, approach, have, want = entry
        lines.append(f"{src}\t{kernel}\t{approach}\t# have {have}/{want}")
    if no_baseline:
        lines.append("#")
        lines.append("# BLOCKED — baseline profile missing for the following "
                     "(src, kernel) entries.")
        lines.append("# Populate via `python benchmark-opt/profile-baselines.py` "
                     "and regenerate the queue.")
        for src, kernel, approach, have, want in no_baseline:
            suffix = f"  # have {have}/{want}" if have else ""
            lines.append(f"# {src}\t{kernel}\t{approach}{suffix}")
    return lines


def _write_queue(path: Path, lines: list[str]) -> None:
    header = [
        "# bench_runs runqueue — generated by `bench-status --runqueue-out`",
        "# Edit freely: comment out lines with '#' or delete them to skip.",
        "# Consume via run-queue.sh.",
        "#",
        "# Each section header looks like:",
        "#   [<cell_name> / <agent_features> / <measurement_method> / <model> / <agent> / <reasoning_effort>]",
        "# <cell_name> is the unique short identifier from matrix.toml. The other",
        "# fields are echoed inline so run-queue.sh doesn't need to consult",
        "# matrix.toml; agent is derived from the agent_features tag.",
        "# Lines under it are tab-separated: source<TAB>kernel<TAB>approach[@variant]",
    ]
    path.write_text("\n".join(header + lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cell", type=str, default=None,
                   help="only report on this matrix.toml cell (matched by `name`)")
    p.add_argument("--runqueue-out", type=Path, default=None,
                   help="write missing/under-replicated cells to this file as an "
                        "editable queue for run-queue.sh")
    args = p.parse_args(argv)
    return report(cell_name=args.cell, runqueue_out=args.runqueue_out)


if __name__ == "__main__":
    sys.exit(main())
