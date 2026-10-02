"""Method-correctness + coverage analysis scoped to two matrix.toml cells.

Scope: the runs described by the `codex_main` (agent_features codex_v2_opt38) and
`codex_lncfix` (codex_v2_opt38_lncfix) cells — model gpt-5.4, the four real-kernel
sources (reference/nl_to_nki/nki_library/nkibench, NOT compiler_gen).

Two outputs:
  1. OFFLINE (this module, no hardware):
     - run-set scoping per (cell, method, kernel), with a substantive/no-op flag
       (does the optimized kernel differ from the baseline in logic, AST-normalized);
     - grounding status (can we partition the input space?);
     - COVERAGE COMPARISON per kernel: which behavior classes the coverage-guided
       generator hits (all, by construction) vs which the kernel's OWN run_test
       property test exercises -> the regimes only coverage-guided generation reaches.
  2. HARDWARE (cell_divergence.py): runs the differential oracle over the generated
     shapes for the substantive+grounded carriers and reports per-class divergences
     (before<->after beyond the optimization's equivalence-class tolerance).

`scope()` returns the run set; `coverage_compare()` the offline table.
"""
from __future__ import annotations

import ast
import csv
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import run_index as ri  # noqa: E402
import grounding as gr  # noqa: E402

_REPO = pathlib.Path(__file__).resolve().parents[2]
_RUNS = _REPO / "benchmark-opt" / "runs"
_MANIFEST = _REPO / "benchmark-opt" / "manifest.csv"

SRC4 = {"reference", "nl_to_nki", "nki_library", "nkibench"}
MAIN_AV = {"no_info@v1", "autocomp@v1", "pre_explored@v1",
           "model_guided_v2@v1", "model_guided_v2@v2", "model_guided_v2@v4"}
LNC_AV = {"no_info@v2", "autocomp@v2", "model_guided_v2@v5", "model_guided_v2@v6"}


def _manifest_features() -> dict[str, str]:
    feat = {}
    if _MANIFEST.exists():
        for r in csv.DictReader(open(_MANIFEST)):
            feat[r["run_dir"].split("/")[-1]] = r.get("agent_features", "")
    return feat


def _cell_of(name: str, feat: dict) -> tuple | None:
    """(cell, approach@variant, source, kernel) or None. v1 variants are shared
    between codex_v1_opt37 and codex_v2_opt38, so require manifest opt38 for those;
    v2/v4/v5/v6 never existed in opt37 so are unambiguous."""
    p = name.split("__")
    if len(p) < 5:
        return None
    src, kern, av, model = p[1], p[2], p[3], p[4]
    if model != "codex_gpt54" or src not in SRC4:
        return None
    if av in MAIN_AV:
        if av.endswith("@v1") and feat.get(name) != "codex_v2_opt38":
            return None
        return ("codex_main", av, src, kern)
    if av in LNC_AV:
        return ("codex_lncfix", av, src, kern)
    return None


def _norm(src: str) -> str:
    """AST dump with docstrings dropped + imports ignored: logic-only signature."""
    try:
        tree = ast.parse(src)
    except Exception as e:  # noqa: BLE001
        return f"PARSEFAIL:{e}"

    class Strip(ast.NodeTransformer):
        def _drop_doc(self, node):
            b = node.body
            if (b and isinstance(b[0], ast.Expr)
                    and isinstance(getattr(b[0], "value", None), ast.Constant)
                    and isinstance(b[0].value.value, str)):
                node.body = b[1:]
            return node

        def visit_Module(self, n):
            self.generic_visit(n)
            self._drop_doc(n)
            n.body = [s for s in n.body if not isinstance(s, (ast.Import, ast.ImportFrom))]
            return n

        def visit_FunctionDef(self, n):
            self.generic_visit(n)
            return self._drop_doc(n)
    tree = Strip().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.dump(tree)


def _substantive(carrier) -> bool:
    return _norm(carrier.baseline_path.read_text()) != _norm(carrier.optimized_path.read_text())


def method_of(av: str) -> str:
    return av.split("@")[0]


def scope():
    """All scoped runs as RunPair-with-metadata dicts."""
    feat = _manifest_features()
    out = []
    for d in sorted(_RUNS.iterdir()):
        if not d.is_dir():
            continue
        c = _cell_of(d.name, feat)
        if c is None:
            continue
        carrier = ri.load_run(d)
        if carrier is None or not carrier.has_pair:
            continue
        cell, av, src, kern = c
        out.append({"cell": cell, "av": av, "method": method_of(av),
                    "source": src, "kernel": kern, "run": d.name,
                    "carrier": carrier})
    return out


def _tested_shapes(carrier):
    """Shapes the oracle's own run_test exercises, in grounding knob order."""
    import kernel_oracle as ko  # noqa: PLC0415
    try:
        omod = ko._load_module(carrier.oracle_path, f"_ts_{carrier.source}_{carrier.kernel}")
    except Exception:  # noqa: BLE001
        return []
    if hasattr(omod, "_make_inputs"):
        return ko.seed_shapes(omod)
    import oracle_recipe as orc  # noqa: PLC0415
    r = orc.parse_recipe(carrier.oracle_path.read_text())
    return list(r.seed_shapes) if (r and r.seed_shapes) else []


def coverage_compare(rep_carrier):
    """For one grounded carrier: the FULL behavior partition (every class with its
    knob/divisor/predicate/critical flag and which generator hit it), plus the actual
    shapes each generator produced. Returns dict or None if ungrounded.

    Accepts either a run_index.RunPair or a scope() record dict (which also carries
    the run dir / cell / approach so the artifact records provenance)."""
    import campaign as cp  # noqa: PLC0415
    rec = rep_carrier if isinstance(rep_carrier, dict) else None
    carrier = rec["carrier"] if rec else rep_carrier
    g = gr.ground_carrier(carrier)
    if not g.grounded:
        return None
    # coverage-guided plan (seed-pinned untiled knobs, as the campaign does)
    try:
        import kernel_oracle as ko  # noqa: PLC0415
        seed = ko.first_seed_shape(carrier)
        seed_vals = (dict(zip(g.knobs, seed))
                     if g.knobs and seed and len(seed) == len(g.knobs) else {})
    except Exception:  # noqa: BLE001
        seed_vals = {}
    plans = cp.plan_shapes(g, seed_vals)
    cg = cp.coverage(g, [p.assignment for p in plans])
    # run_test (property-test) coverage against the SAME partition
    shapes = _tested_shapes(carrier)
    assigns = [dict(zip(g.knobs, s)) for s in shapes if len(s) == len(g.knobs)]
    rt = cp.coverage(g, assigns)
    cg_hit_names, rt_hit_names = set(cg.class_hits), set(rt.class_hits)

    classes = [{
        "name": c.name, "knob": c.knob, "divisor": c.divisor,
        "predicate": c.predicate, "critical": c.critical,
        "hit_by_coverage_guided": c.name in cg_hit_names,
        "hit_by_run_test": c.name in rt_hit_names,
    } for c in g.classes]

    out = {
        "kernel": f"{g.source}/{g.kernel}",
        "knobs": g.knobs,
        "tile_divisors": g.tile_divisors,
        "n_classes": len(g.classes),
        "classes": classes,                       # the ACTUAL derived behavior classes
        "coverage_guided_shapes": [list(p.shape) for p in plans],
        "run_test_shapes": [list(s) for s in shapes],
        "cg_hit": cg.bins_hit, "cg_shapes": len(plans),
        "rt_hit": rt.bins_hit, "rt_shapes": len(assigns),
        "rt_missing": sorted(set(c.name for c in g.classes) - rt_hit_names),
        "critical": [c.name for c in g.classes if c.critical],
    }
    if rec:
        out["representative_run"] = rec["run"]
        out["cell"] = rec["cell"]
        out["approach"] = rec["av"]
    return out


def coverage_compare_all(runs=None):
    """Coverage comparison (with full behavior classes) for every in-scope kernel.
    Picks one representative per distinct kernel, preferring a substantive run.
    Ungrounded kernels are included with their reason so the artifact is complete."""
    import collections  # noqa: PLC0415
    runs = runs if runs is not None else scope()
    by_kernel = collections.defaultdict(list)
    for r in runs:
        by_kernel[(r["source"], r["kernel"])].append(r)
    rows = []
    for key in sorted(by_kernel):
        rep = next((r for r in by_kernel[key] if _substantive(r["carrier"])),
                   by_kernel[key][0])
        cc = coverage_compare(rep)
        if cc is None:
            g = gr.ground_carrier(rep["carrier"])
            rows.append({"kernel": f"{key[0]}/{key[1]}", "grounded": False,
                         "ungrounded_reason": g.ungrounded[0] if g.ungrounded else "",
                         "representative_run": rep["run"],
                         "cell": rep["cell"], "approach": rep["av"]})
        else:
            cc["grounded"] = True
            rows.append(cc)
    return rows


if __name__ == "__main__":
    import collections
    import json
    runs = scope()
    by_kernel = collections.defaultdict(list)
    for r in runs:
        by_kernel[(r["source"], r["kernel"])].append(r)

    print(f"# scope: {len(runs)} runs, {len(by_kernel)} distinct kernels "
          f"(codex_main + codex_lncfix, model gpt-5.4)\n")

    print("## run counts per cell / method")
    agg = collections.Counter()
    sub = collections.Counter()
    for r in runs:
        agg[(r["cell"], r["method"])] += 1
        if _substantive(r["carrier"]):
            sub[(r["cell"], r["method"])] += 1
    for k in sorted(agg):
        print(f"  {k[0]:13s} {k[1]:16s} runs={agg[k]:3d}  substantive={sub[k]:3d}")

    rows = coverage_compare_all(runs)
    out_path = pathlib.Path(__file__).parent / "cell_coverage_compare.json"
    out_path.write_text(json.dumps(rows, indent=2) + "\n")
    gnd = [r for r in rows if r.get("grounded")]
    tot_def = sum(r["n_classes"] for r in gnd)
    tot_cg = sum(r["cg_hit"] for r in gnd)
    tot_rt = sum(r["rt_hit"] for r in gnd)
    print(f"\n## coverage: grounded={len(gnd)}/{len(rows)} kernels | classes "
          f"defined={tot_def} coverage_guided={tot_cg} run_test={tot_rt}")
    print(f"[wrote {out_path} — full behavior classes per kernel]")

    print("\n## coverage comparison (coverage-guided vs original run_test), per kernel")
    print(f"  {'kernel':28s} {'grnd':4s} {'#cls':4s} {'cg':>5s} {'run_test':>8s}  missing-by-run_test")
    for r in rows:
        if not r.get("grounded"):
            print(f"  {r['kernel']:28s}  no")
            continue
        miss = ",".join(r["rt_missing"]) or "—"
        print(f"  {r['kernel']:28s} yes  {r['n_classes']:>3d}  "
              f"{r['cg_hit']:>2d}/{r['n_classes']:<2d} {r['rt_hit']:>3d}/{r['n_classes']:<2d}   {miss}")
