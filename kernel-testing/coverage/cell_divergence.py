"""Per-run divergence detection over coverage-guided shapes (needs venv + device).

For each substantive+grounded carrier in the codex_main / codex_lncfix scope, ground
the input space, generate the covering shapes, run the BEFORE and AFTER kernels on
each, and classify every behavior class by the three differentials:

  b<->a  (before vs after)  : did the optimization change the output here?
  b<->ref(before vs torch)  : was the baseline correct here?
  a<->ref(after  vs torch)  : is the optimized kernel correct here?

A DIVERGENCE is a class where b<->a is non-trivial (exceeds bit-exact). We grade it:
  - benign  : b<->a within the fp-reorder tier AND optimized still correct (a<->ref ok)
  - REGRESSION: optimized is wrong (a<->ref exceeds the kernel's own oracle tolerance)
                while the baseline was correct (b<->ref ok) — the method broke a regime.

All tolerances are magnitude-scaled (err <= atol + rtol*ref_absmax), matching the
campaign. The kernel's own oracle atol/rtol is used for correctness; the _TOL tiers
for the equivalence grade. Lncfix runs lack analysis.json, so we don't rely on
opt-ids — we report the raw magnitudes and the tier each crosses.
"""
from __future__ import annotations

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cell_analysis as ca  # noqa: E402
import grounding as gr  # noqa: E402
import campaign as cp  # noqa: E402

# the 4 substantive+grounded carriers (cell, source, kernel)
WORKLIST = [
    ("codex_lncfix", "reference", "layernorm"),
    ("codex_lncfix", "reference", "rms"),
    ("codex_lncfix", "reference", "rope"),
    ("codex_main", "nl_to_nki", "layernorm"),
]

_TIERS = ["A_bitexact", "B_fp_reorder", "C_precision"]


def _tier_crossed(err, mag):
    """Loosest tier the error stays WITHIN, or 'EXCEEDS_ALL'."""
    if err is None:
        return "n/a"
    for t in _TIERS:
        a, r = cp._TOL[t]
        if err <= a + r * mag:
            return t
    return "EXCEEDS_ALL"


def analyze(carrier):
    import kernel_oracle as ko  # noqa: PLC0415
    import oracle_recipe as orc  # noqa: PLC0415

    g = gr.ground_carrier(carrier)
    baseline_ok = ko.baseline_run_test_ok(carrier)
    seed = ko.first_seed_shape(carrier)
    seed_vals = (dict(zip(g.knobs, seed))
                 if g.knobs and seed and len(seed) == len(g.knobs) else {})
    plans = cp.plan_shapes(g, seed_vals)
    o_atol, o_rtol = orc.assert_close_tol(carrier.oracle_path.read_text())
    o_atol = o_atol if o_atol is not None else 1e-2
    o_rtol = o_rtol if o_rtol is not None else 1e-2
    crit = {c.name for c in g.classes if c.critical}

    classes = []
    for p in plans:
        res = ko.run_pair_for(carrier, p.shape, seed=42)
        mag = res.ref_absmax if (res.ran and res.ref_absmax is not None) else 1.0
        for cname in p.targets:
            if not res.ran:
                classes.append({"class": cname, "shape": list(p.shape),
                                "ran": False, "detail": res.detail})
                continue
            ba, br, ar = res.before_vs_after, res.before_vs_ref, res.after_vs_ref
            otol = o_atol + o_rtol * mag
            base_correct = br is not None and br <= otol
            opt_correct = ar is not None and ar <= otol
            eq_tier = _tier_crossed(ba, mag)
            diverged = ba is not None and ba > cp._TOL["A_bitexact"][0] + cp._TOL["A_bitexact"][1] * mag
            grade = "identical"
            if diverged:
                if base_correct and not opt_correct:
                    grade = "REGRESSION"          # method broke a correct regime
                elif not base_correct and not opt_correct:
                    grade = "both-wrong"
                elif eq_tier in ("EXCEEDS_ALL", "C_precision"):
                    grade = "large-but-correct"
                else:
                    grade = "benign-reorder"
            classes.append({
                "class": cname, "critical": cname in crit, "shape": list(p.shape),
                "ran": True, "b_vs_a": ba, "b_vs_ref": br, "a_vs_ref": ar,
                "ref_absmax": mag, "eq_tier": eq_tier,
                "baseline_correct": base_correct, "optimized_correct": opt_correct,
                "grade": grade,
                "detail": f"b<->a={ba:.4g} b<->ref={br:.4g} a<->ref={ar:.4g}",
            })
    return {
        "kernel": f"{g.source}/{g.kernel}",
        "baseline_passes_run_test": baseline_ok,
        "oracle_tol": [o_atol, o_rtol],
        "n_classes": len(g.classes),
        "classes": classes,
    }


def main():
    runs = ca.scope()
    out_dir = HERE.parent / "findings"
    results = []
    for cell, src, kern in WORKLIST:
        r = next((x for x in runs if x["cell"] == cell and x["source"] == src
                  and x["kernel"] == kern and ca._substantive(x["carrier"])), None)
        if r is None:
            print(f"!! no substantive run for {cell} {src}/{kern}")
            continue
        print(f"\n===== {cell}  {src}/{kern}  ({r['av']}) =====", flush=True)
        res = analyze(r["carrier"])
        res["cell"], res["approach"], res["run"] = cell, r["av"], r["run"]
        results.append(res)
        print(f"  baseline_run_test={res['baseline_passes_run_test']}  "
              f"oracle_tol={res['oracle_tol']}")
        for c in res["classes"]:
            if not c["ran"]:
                print(f"    {c['class']:22s} {str(c['shape']):16s} DID-NOT-RUN: {c['detail']}")
                continue
            mark = {"identical": "  ", "benign-reorder": "  ",
                    "large-but-correct": " ~", "REGRESSION": " <-- REGRESSION",
                    "both-wrong": " <-- BOTH-WRONG"}.get(c["grade"], "")
            print(f"    {c['class']:22s} {str(c['shape']):16s} {c['grade']:18s} "
                  f"{c['detail']}{mark}")
    (HERE / "cell_divergence_results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"\n[wrote {HERE/'cell_divergence_results.json'}]")


if __name__ == "__main__":
    main()
