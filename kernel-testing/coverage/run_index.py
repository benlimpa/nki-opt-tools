"""Index the optimization runs into (optimization -> real carrier kernels).

The coverage metric is keyed on the ~24 named optimizations, and the *carriers*
of each optimization are the real benchmark kernels it was actually applied to.
Those live as before/after pairs under `benchmark-opt/runs/<run>/workspace/`:

    workspace/kernel.py            # baseline (pre-optimization)
    workspace/optimized_kernel.py  # optimized (post-optimization)
    workspace/oracle.py            # shared _make_inputs / _ref

and the optimizations applied are recorded in `<run>/analysis.json` at
`analysis.optimizations_selected_ids` (a list-of-lists, one inner list per
optimization step; we flatten + unique it).

Only a subset of runs carry `analysis.json` (the opt-id labelling comes from a
post-processing pass). EVERY run has the before/after pair, so the labelling --
not the pair -- is the limiting factor. `index_runs` reports both so the
unlabelled remainder (a future expansion: recover ids from report.md/run.jsonl)
is never silently dropped.

Pure stdlib (json/os/pathlib) -- no torch, no device. Safe to import in the
offline test interpreter.
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field

# repo_root/benchmark-opt/runs  (this file is repo_root/kernel-testing/coverage/)
_REPO = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_RUNS_DIR = _REPO / "benchmark-opt" / "runs"


@dataclass
class RunPair:
    """One before/after optimization run with its applied optimization ids."""
    run_dir: pathlib.Path
    source: str
    kernel: str
    approach: str
    variant: str
    model: str
    opt_ids: list[int] = field(default_factory=list)   # flattened + unique
    labelled: bool = False        # analysis.json present with opt ids
    success: bool | None = None   # analysis.final_optimization_successful

    @property
    def baseline_path(self) -> pathlib.Path:
        return self.run_dir / "workspace" / "kernel.py"

    @property
    def optimized_path(self) -> pathlib.Path:
        return self.run_dir / "workspace" / "optimized_kernel.py"

    @property
    def oracle_path(self) -> pathlib.Path:
        return self.run_dir / "workspace" / "oracle.py"

    @property
    def has_pair(self) -> bool:
        return (self.baseline_path.exists()
                and self.optimized_path.exists()
                and self.oracle_path.exists())


def parse_run_name(name: str) -> dict | None:
    """`TIMESTAMP__source__kernel__approach@variant__model` -> dict, or None.

    Returns None for names that don't match the 5-segment convention rather
    than guessing, so malformed dirs are reported, not silently mis-bucketed."""
    parts = name.split("__")
    if len(parts) != 5:
        return None
    _ts, source, kernel, approach_variant, model = parts
    if "@" in approach_variant:
        approach, variant = approach_variant.split("@", 1)
    else:
        approach, variant = approach_variant, ""
    return {"source": source, "kernel": kernel,
            "approach": approach, "variant": variant, "model": model}


def _flatten_opt_ids(raw) -> list[int]:
    """`[[31], [], [20]]` -> `[31, 20]` (flattened, unique, order-preserving).

    Tolerates already-flat lists and stray non-ints."""
    out: list[int] = []
    if not isinstance(raw, list):
        return out
    for grp in raw:
        items = grp if isinstance(grp, list) else [grp]
        for x in items:
            if isinstance(x, bool):          # bool is an int subclass; reject
                continue
            if isinstance(x, int) and x not in out:
                out.append(x)
    return out


def load_run(run_dir: pathlib.Path) -> RunPair | None:
    """Build a RunPair from a run directory. Returns None if the dir name is
    malformed. A run with no analysis.json is still returned (labelled=False,
    opt_ids=[]) because its before/after pair is usable for other purposes."""
    meta = parse_run_name(run_dir.name)
    if meta is None:
        return None
    pair = RunPair(run_dir=run_dir, **meta)
    ap = run_dir / "analysis.json"
    if ap.exists():
        try:
            a = json.loads(ap.read_text())
        except (ValueError, OSError):
            a = None
        if isinstance(a, dict):
            an = a.get("analysis")
            if isinstance(an, dict):
                ids = _flatten_opt_ids(an.get("optimizations_selected_ids"))
                pair.opt_ids = ids
                pair.labelled = an.get("optimizations_selected_ids") is not None
                pair.success = an.get("final_optimization_successful")
    return pair


@dataclass
class IndexReport:
    pairs: list[RunPair]                 # all parseable runs
    labelled: list[RunPair]              # subset with opt ids
    by_opt: dict[int, list[RunPair]]     # opt id -> carriers (labelled, has_pair)
    malformed: list[str]                 # run dir names that didn't parse

    @property
    def unlabelled(self) -> list[RunPair]:
        return [p for p in self.pairs if not p.labelled]

    def summary(self) -> str:
        lines = [
            f"runs scanned:     {len(self.pairs) + len(self.malformed)}",
            f"  parseable:      {len(self.pairs)}",
            f"  malformed name: {len(self.malformed)}",
            f"labelled (opt ids): {len(self.labelled)}",
            f"unlabelled pairs:   {len(self.unlabelled)}  "
            f"(have before/after but no opt-id; future expansion)",
            f"optimizations with >=1 carrier: {len(self.by_opt)}",
        ]
        for opt in sorted(self.by_opt):
            cs = self.by_opt[opt]
            kernels = sorted({f"{c.source}/{c.kernel}" for c in cs})
            lines.append(f"  opt{opt:02d}: {len(cs)} run(s) over "
                         f"{len(kernels)} kernel(s): {', '.join(kernels)}")
        return "\n".join(lines)


def index_runs(runs_dir: pathlib.Path = DEFAULT_RUNS_DIR) -> IndexReport:
    """Scan a runs directory and invert to opt id -> carriers."""
    pairs: list[RunPair] = []
    malformed: list[str] = []
    if runs_dir.is_dir():
        for child in sorted(runs_dir.iterdir()):
            if not child.is_dir():
                continue
            p = load_run(child)
            if p is None:
                malformed.append(child.name)
            else:
                pairs.append(p)
    labelled = [p for p in pairs if p.labelled and p.opt_ids]
    by_opt: dict[int, list[RunPair]] = {}
    for p in labelled:
        if not p.has_pair:           # labelled but pair files missing -> skip as carrier
            continue
        for opt in p.opt_ids:
            by_opt.setdefault(opt, []).append(p)
    return IndexReport(pairs=pairs, labelled=labelled, by_opt=by_opt,
                       malformed=malformed)


if __name__ == "__main__":
    import sys
    rd = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_RUNS_DIR
    print(f"runs dir: {rd}\n")
    print(index_runs(rd).summary())
