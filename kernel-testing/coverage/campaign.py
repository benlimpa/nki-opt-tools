"""Coverage campaign: opt -> carriers -> ground -> generate -> run_pair -> tally.

For each optimization and each carrier it was applied to, this:

  1. grounds the carrier into a divisibility behavior-class partition (grounding.py);
  2. plans a minimal shape set covering the partition -- one all-exact base shape
     (covers every `*_exact` class at once) plus one targeted shape per critical
     class (the violated knob set just over / just under its tile divisor);
  3. [device] runs the before/after oracle on each shape (kernel_oracle.run_pair);
  4. tallies coverage (bins_hit / bins_defined, uncovered criticals surfaced) and
     classifies every covered class as equivalence_preserved (before<->after within
     tolerance) and baseline_in_domain (before<->ref small).

Steps 1-2 and the tally are pure (offline-testable). Step 3 needs the Neuron venv
+ device. The equivalence/in-domain split is what keeps the metric honest: a class
where before<->after==0 but neither matches torch is reported as equivalence-
preserved-but-out-of-contract, NOT as "tested correct"
(see findings/equivalence-necessary-not-sufficient.md).

Cost note: scan-style kernels (e.g. nkibench/mamba) recompile a NEFF per scan
step, so one shape can trigger hundreds of compiles. Run such carriers in the
background / one opt at a time; the cheap matmul-family carriers (adamw, matmul,
lora, ...) compile two NEFFs per shape and run in seconds.

Validated on hardware (opt31 / nkibench/adamw):
    nkibench/adamw: 3/3 bins (100%)
        N_x2048_exact    (256, 4096)  [eq /in-dom] b<->a=0 b<->ref=4.8e-07
      ! N_x2048_partial  (256, 2049)  [eq /OUT   ] b<->a=0 b<->ref=3.445
      ! N_x2048_below    (256, 1024)  [eq /OUT   ] b<->a=0 b<->ref=6.249
opt31 preserves equivalence in every class; baseline is in-domain only at exact.
"""
from __future__ import annotations

import pathlib
import sys
from dataclasses import dataclass, field

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import grounding as gr  # noqa: E402
import run_index as ri  # noqa: E402

DEFAULT_UNTILED = 256   # value for knobs with no tile divisor (kept fixed/valid)


# ---------------------------------------------------------------- shape planning (pure)

@dataclass
class ShapePlan:
    shape: tuple              # in oracle-knob order (ready for _make_inputs(*shape))
    assignment: dict          # knob -> value
    targets: list[str]        # class names this shape was built to hit


def _base_assignment(g: gr.CarrierGrounding, seed_vals: dict | None = None) -> dict:
    """All-exact base: tiled knobs -> max divisor (one full tile, knob%D==0 for
    power-of-two tiles); untiled knobs -> their value from the oracle's first seed
    shape (a tested, known-good value) if available, else DEFAULT_UNTILED. Pinning
    untiled knobs to a tested value keeps OOD attributable to the tiled axis under
    test (not an arbitrary default, e.g. lora R)."""
    seed_vals = seed_vals or {}
    a = {}
    for knob in g.knobs:
        divs = g.tile_divisors.get(knob)
        if divs:
            a[knob] = max(divs)
        else:
            a[knob] = seed_vals.get(knob, DEFAULT_UNTILED)
    return a


def _violating_value(regime: str, D: int) -> int:
    if regime == "partial":          # > D, remainder != 0
        return D + 1
    if regime == "below":            # 0 < v < D, zero full tiles
        return max(1, D // 2)
    return 2 * D                     # exact


def _regime_of(class_name: str) -> str:
    return class_name.rsplit("_", 1)[-1]   # ..._exact|_partial|_below


def plan_shapes(g: gr.CarrierGrounding, seed_vals: dict | None = None) -> list[ShapePlan]:
    """Covering plan: one all-exact base (covers every *_exact class at once) plus
    one targeted shape per non-exact class (partial/below), regardless of whether
    it is critical -- we want to exercise padded tails too, not just bug-likely
    ones. The base targets ONLY the exact classes it actually satisfies."""
    if not g.grounded:
        return []
    knobs = g.knobs
    base = _base_assignment(g, seed_vals)
    exact = [c.name for c in g.classes if _regime_of(c.name) == "exact"]
    plans = [ShapePlan(tuple(base[k] for k in knobs), dict(base), exact)]
    for c in g.classes:
        if _regime_of(c.name) == "exact":
            continue
        a = dict(base)
        a[c.knob] = _violating_value(_regime_of(c.name), c.divisor)
        plans.append(ShapePlan(tuple(a[k] for k in knobs), a, [c.name]))
    return plans


# ---------------------------------------------------------------- coverage tally (pure)

def _matches(pred: str, assignment: dict) -> bool:
    try:
        return bool(eval(pred, {"__builtins__": {}}, assignment))  # noqa: S307
    except Exception:  # noqa: BLE001
        return False


@dataclass
class CoverageReport:
    source: str
    kernel: str
    bins_defined: int
    bins_hit: int
    uncovered: list[str] = field(default_factory=list)
    uncovered_critical: list[str] = field(default_factory=list)
    class_hits: dict = field(default_factory=dict)   # class name -> [assignments hitting it]

    @property
    def fraction(self) -> float:
        return self.bins_hit / self.bins_defined if self.bins_defined else 1.0

    def summary(self) -> str:
        head = (f"{self.source}/{self.kernel}: "
                f"{self.bins_hit}/{self.bins_defined} bins "
                f"({self.fraction*100:.0f}%)")
        if self.uncovered_critical:
            head += f"  UNCOVERED CRITICAL: {', '.join(self.uncovered_critical)}"
        elif self.uncovered:
            head += f"  uncovered: {', '.join(self.uncovered)}"
        return head


def coverage(g: gr.CarrierGrounding, assignments: list[dict]) -> CoverageReport:
    """Which behavior classes are hit by a set of knob assignments."""
    hits: dict[str, list[dict]] = {}
    for c in g.classes:
        for a in assignments:
            if all(k in a for k in g.knobs) and _matches(c.predicate, a):
                hits.setdefault(c.name, []).append(a)
    defined = [c.name for c in g.classes]
    uncovered = [n for n in defined if n not in hits]
    crit = {c.name for c in g.classes if c.critical}
    return CoverageReport(
        source=g.source, kernel=g.kernel,
        bins_defined=len(defined), bins_hit=len(hits),
        uncovered=uncovered,
        uncovered_critical=[n for n in uncovered if n in crit],
        class_hits=hits,
    )


# ---------------------------------------------------------------- equivalence-class tolerance

# equivalence_class -> (atol, rtol) for the before<->after assertion and the
# before<->ref in-domain check. Tiling opts are bit-exact; FP-reorder loosens.
_TOL = {
    "A_bitexact": (1e-5, 1e-5),
    "B_fp_reorder": (1e-2, 1e-2),
    "C_precision": (5e-2, 5e-2),
}


def _equivalence_class_for(opt: int, spec_path: pathlib.Path) -> str:
    """Read the opt's equivalence_class from coverage_spec.toml; default bitexact."""
    try:
        import tomllib  # noqa: PLC0415
        spec = tomllib.loads(spec_path.read_text())
    except Exception:  # noqa: BLE001
        return "A_bitexact"
    return spec.get(f"opt{opt:02d}", {}).get("equivalence_class", "A_bitexact")


# ---------------------------------------------------------------- device campaign

@dataclass
class ClassVerdict:
    class_name: str
    critical: bool
    shape: tuple
    equivalence_preserved: bool | None    # before<->after within tol
    baseline_in_domain: bool | None       # before<->ref small
    detail: str


def run_carrier(carrier, opt: int, *, spec_path: pathlib.Path, seed: int = 42,
                gate: bool = True):
    """Ground + plan + run_pair a single carrier. Returns (grounding, coverage,
    per-class verdicts, baseline_ok). baseline_ok is the run_test sanity gate:
    True/False/None for whether the baseline passes its own oracle.run_test (a
    False means the run's baseline is already broken — its verdicts aren't
    trustworthy as findings). Imports the device oracle lazily (needs the venv)."""
    import oracle_recipe as orc  # noqa: PLC0415
    import kernel_oracle as ko  # noqa: PLC0415

    g = gr.ground_carrier(carrier)
    baseline_ok = ko.baseline_run_test_ok(carrier) if gate else None
    seed_shape = ko.first_seed_shape(carrier)
    seed_vals = dict(zip(g.knobs, seed_shape)) if (g.knobs and seed_shape
                     and len(seed_shape) == len(g.knobs)) else {}
    plans = plan_shapes(g, seed_vals)
    cov = coverage(g, [p.assignment for p in plans])

    # Two distinct tolerances, magnitude-scaled (~torch.testing.assert_close):
    #   equivalence (before<->after): the opt's equivalence-class tolerance;
    #   in-domain   (before<->ref):   the kernel's OWN oracle tolerance.
    eq_atol, eq_rtol = _TOL.get(_equivalence_class_for(opt, spec_path), _TOL["A_bitexact"])
    o_atol, o_rtol = orc.assert_close_tol(carrier.oracle_path.read_text())
    ref_atol = o_atol if o_atol is not None else 1e-2
    ref_rtol = o_rtol if o_rtol is not None else 1e-2

    verdicts: list[ClassVerdict] = []
    crit_names = {c.name for c in g.classes if c.critical}
    for p in plans:
        res = ko.run_pair_for(carrier, p.shape, seed=seed)
        mag = res.ref_absmax if (res.ran and res.ref_absmax is not None) else 1.0
        for cname in p.targets:
            eq = indom = None
            if res.ran:
                ba, br = res.before_vs_after, res.before_vs_ref
                eq = ba is not None and ba <= eq_atol + eq_rtol * mag
                indom = br is not None and br <= ref_atol + ref_rtol * mag
            verdicts.append(ClassVerdict(
                cname, cname in crit_names, p.shape, eq, indom,
                res.detail if not res.ran else str(res)))
    return g, cov, verdicts, baseline_ok


@dataclass
class CarrierRow:
    opt: int
    source: str
    kernel: str
    grounded: bool
    bins_hit: int
    bins_defined: int
    n_eq: int                 # classes with equivalence_preserved
    n_indomain: int           # classes with baseline_in_domain
    n_classes: int            # classes actually run
    out_of_domain: list       # (class, shape, before_vs_ref) where baseline is OUT
    baseline_ok: bool | None = None   # passes its own run_test?
    note: str = ""

    @property
    def cov(self) -> str:
        return f"{self.bins_hit}/{self.bins_defined}" if self.grounded else "-"


def _table(rows: list) -> str:
    out = ["| opt | carrier | baseline | coverage | eq | in-domain | out-of-contract bins |",
           "|---|---|---|---|---|---|---|"]
    for r in rows:
        if not r.grounded:
            out.append(f"| opt{r.opt:02d} | {r.source}/{r.kernel} | | _ungrounded_ | | | {r.note} |")
            continue
        base = {True: "ok", False: "**FAILS run_test**", None: "?"}[r.baseline_ok]
        oods = ", ".join(f"{c}{s}" for c, s, _ in r.out_of_domain) or "—"
        out.append(f"| opt{r.opt:02d} | {r.source}/{r.kernel} | {base} | {r.cov} | "
                   f"{r.n_eq}/{r.n_classes} | {r.n_indomain}/{r.n_classes} | {oods} |")
    return "\n".join(out)


def run_campaign(opts: list[int] | None = None, *,
                 runs_dir: pathlib.Path = ri.DEFAULT_RUNS_DIR,
                 spec_path: pathlib.Path | None = None, seed: int = 42,
                 skip_kernels: frozenset = frozenset(),
                 table_path: pathlib.Path | None = None) -> list:
    if spec_path is None:
        spec_path = pathlib.Path(__file__).parent / "coverage_spec.toml"
    rep = ri.index_runs(runs_dir)
    target_opts = sorted(rep.by_opt) if opts is None else [o for o in opts if o in rep.by_opt]
    rows: list[CarrierRow] = []
    for opt in target_opts:
        seen: dict[tuple, object] = {}
        for c in rep.by_opt[opt]:
            seen.setdefault((c.source, c.kernel), c)
        print(f"\n========== opt{opt:02d}  ({len(seen)} carrier kernel(s)) ==========", flush=True)
        for key in sorted(seen):
            if key[1] in skip_kernels:
                print(f"  SKIP {key[0]}/{key[1]} (skip-list)", flush=True)
                continue
            g, cov, verdicts, baseline_ok = run_carrier(seen[key], opt, spec_path=spec_path, seed=seed)
            if not g.grounded:
                print(f"  {g.summary()}", flush=True)
                rows.append(CarrierRow(opt, key[0], key[1], False, 0, 0, 0, 0, 0, [],
                                       note=(g.ungrounded[0] if g.ungrounded else "")))
                continue
            ran = [v for v in verdicts if v.equivalence_preserved is not None]
            ood = [(v.class_name, v.shape, None) for v in ran if v.baseline_in_domain is False]
            rows.append(CarrierRow(
                opt, key[0], key[1], True, cov.bins_hit, cov.bins_defined,
                sum(v.equivalence_preserved is True for v in ran),
                sum(v.baseline_in_domain is True for v in ran),
                len(ran), ood, baseline_ok=baseline_ok))
            base_str = {True: "", False: "  [BASELINE FAILS run_test]", None: ""}[baseline_ok]
            print(f"  {cov.summary()}{base_str}", flush=True)
            for v in verdicts:
                flag = "!" if v.critical else " "
                eq = {True: "eq", False: "NEQ", None: "?"}[v.equivalence_preserved]
                dom = {True: "in-dom", False: "OUT", None: "?"}[v.baseline_in_domain]
                print(f"    {flag} {v.class_name:22s} {str(v.shape):20s} "
                      f"[{eq:3s}/{dom:6s}] {v.detail}", flush=True)
    table = _table(rows)
    print("\n\n===== COVERAGE TABLE =====\n" + table, flush=True)
    if table_path is not None:
        table_path.write_text(table + "\n")
        print(f"\n[wrote table to {table_path}]", flush=True)
    return rows


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("opts", nargs="*", type=int, help="optimization ids (default: all)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip", default="", help="comma-separated kernel names to skip")
    ap.add_argument("--table", default="", help="path to write the markdown table")
    args = ap.parse_args()
    run_campaign(args.opts or None, seed=args.seed,
                 skip_kernels=frozenset(k for k in args.skip.split(",") if k),
                 table_path=pathlib.Path(args.table) if args.table else None)
