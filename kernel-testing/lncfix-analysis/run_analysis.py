"""Coverage + divergence analysis for the codex_lncfix cell (matrix.toml).

Self-contained driver that saves ALL artifacts under this directory:

  coverage/coverage_compare.json     aggregate: per-kernel behavior partition,
                                     coverage-guided vs run_test hits, missed bins
  coverage/by_kernel/<src>__<k>.json one file per grounded kernel (full classes +
                                     the actual coverage-guided / run_test shapes)
  coverage/missed_bins.json          every bin the property tests miss, with the
                                     coverage-guided shape that reaches it
  divergence/divergence_results.json aggregate: per-(run,class) b<->a / b<->ref /
                                     a<->ref + grade, over coverage-guided shapes
  divergence/by_run/<run>.json       one file per substantive+grounded run
  summary.json                       scope counts + headline coverage/divergence totals

Coverage is OFFLINE (AST only). Divergence needs the Neuron venv + a trn1 device;
without it, the divergence step records why it skipped and leaves prior results intact.

Reuses coverage/cell_analysis.py (scoping + coverage) and coverage/cell_divergence.py
(per-run differential). Run:
    python3 kernel-testing/lncfix-analysis/run_analysis.py                 # coverage only
    PATH=$VENV/bin:$PATH NEURON_PLATFORM_TARGET_OVERRIDE=trn1 \
        $VENV/bin/python kernel-testing/lncfix-analysis/run_analysis.py    # + divergence
"""
from __future__ import annotations

import collections
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
COV = HERE.parent / "coverage"
sys.path.insert(0, str(COV))

import cell_analysis as ca  # noqa: E402
import grounding as gr  # noqa: E402

CELL = "codex_lncfix"


def _covering_shape(rec, cls):
    """A coverage-guided shape (in knob order) whose assignment satisfies the class."""
    for shp in rec["coverage_guided_shapes"]:
        a = dict(zip(rec["knobs"], shp))
        try:
            if eval(cls["predicate"], {"__builtins__": {}}, a):  # noqa: S307
                return shp
        except Exception:  # noqa: BLE001
            pass
    return None


def do_coverage(runs):
    """Offline: per distinct kernel, the behavior partition + coverage comparison."""
    by_kernel = collections.defaultdict(list)
    for r in runs:
        by_kernel[(r["source"], r["kernel"])].append(r)
    rows = []
    for key in sorted(by_kernel):
        rep = next((r for r in by_kernel[key] if ca._substantive(r["carrier"])),
                   by_kernel[key][0])
        cc = ca.coverage_compare(rep)
        if cc is None:
            g = gr.ground_carrier(rep["carrier"])
            rows.append({"kernel": f"{key[0]}/{key[1]}", "grounded": False,
                         "ungrounded_reason": g.ungrounded[0] if g.ungrounded else "",
                         "representative_run": rep["run"], "approach": rep["av"]})
        else:
            cc["grounded"] = True
            rows.append(cc)
            (HERE / "coverage" / "by_kernel"
             / f"{key[0]}__{key[1]}.json").write_text(json.dumps(cc, indent=2) + "\n")
    (HERE / "coverage" / "coverage_compare.json").write_text(json.dumps(rows, indent=2) + "\n")

    # missed-bin breakdown
    missed = []
    for r in rows:
        if not r.get("grounded"):
            continue
        for c in r["classes"]:
            if not c["hit_by_run_test"]:
                missed.append({"kernel": r["kernel"], "approach": r.get("approach"),
                               "class": c["name"], "critical": c["critical"],
                               "predicate": c["predicate"],
                               "covered_by_coverage_guided": c["hit_by_coverage_guided"],
                               "coverage_guided_shape": _covering_shape(r, c),
                               "run_test_shapes": r["run_test_shapes"]})
    (HERE / "coverage" / "missed_bins.json").write_text(json.dumps(missed, indent=2) + "\n")
    return rows, missed


def do_divergence(runs):
    """On device: per substantive+grounded run, the per-class differential."""
    try:
        import cell_divergence as cd  # noqa: PLC0415
        import kernel_oracle  # noqa: F401, PLC0415
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"
    work = [r for r in runs
            if ca._substantive(r["carrier"]) and gr.ground_carrier(r["carrier"]).grounded]
    results = []
    for r in sorted(work, key=lambda r: (r["av"], r["source"], r["kernel"])):
        print(f"  [div] {r['av']:22s} {r['source']}/{r['kernel']}", flush=True)
        try:
            res = cd.analyze(r["carrier"])
        except Exception as e:  # noqa: BLE001
            res = {"kernel": f"{r['source']}/{r['kernel']}", "error": f"{type(e).__name__}: {e}"}
        res["run"], res["approach"] = r["run"], r["av"]
        results.append(res)
        (HERE / "divergence" / "by_run" / f"{r['run']}.json").write_text(
            json.dumps(res, indent=2) + "\n")
    (HERE / "divergence" / "divergence_results.json").write_text(
        json.dumps(results, indent=2) + "\n")
    return results, None


def main():
    runs = [r for r in ca.scope() if r["cell"] == CELL]
    agg = collections.Counter()
    sub = collections.Counter()
    for r in runs:
        agg[r["method"]] += 1
        if ca._substantive(r["carrier"]):
            sub[r["method"]] += 1
    print(f"# {CELL}: {len(runs)} runs with pair")
    for m in sorted(agg):
        print(f"  {m:16s} runs={agg[m]:3d} substantive={sub[m]:3d}")

    rows, missed = do_coverage(runs)
    gnd = [r for r in rows if r.get("grounded")]
    tot_def = sum(r["n_classes"] for r in gnd)
    tot_cg = sum(r["cg_hit"] for r in gnd)
    tot_rt = sum(r["rt_hit"] for r in gnd)
    print(f"\n[coverage] grounded={len(gnd)}/{len(rows)} kernels | classes defined={tot_def} "
          f"coverage_guided={tot_cg} run_test={tot_rt} | missed_by_run_test={len(missed)} "
          f"({sum(m['critical'] for m in missed)} critical)")

    results, skip = do_divergence(runs)
    summary = {
        "cell": CELL, "runs_with_pair": len(runs),
        "runs_by_method": dict(agg), "substantive_by_method": dict(sub),
        "coverage": {"grounded_kernels": len(gnd), "total_kernels": len(rows),
                     "classes_defined": tot_def, "coverage_guided_hit": tot_cg,
                     "run_test_hit": tot_rt, "bins_missed_by_run_test": len(missed),
                     "critical_bins_missed": sum(m["critical"] for m in missed)},
    }
    if skip:
        print(f"[divergence] SKIPPED — {skip}")
        summary["divergence"] = {"ran": False, "reason": skip}
    else:
        regr = [(d["run"], d["kernel"], c) for d in results for c in d.get("classes", [])
                if c.get("grade") == "REGRESSION"]
        # the class-level flag count double-counts the all-exact base shape (which
        # satisfies several *_exact predicates), so also report deduped kernels/shapes.
        regr_kernels = sorted({k for _, k, _ in regr})
        regr_shapes = sorted({(k, tuple(c["shape"])) for _, k, c in regr})
        print(f"[divergence] tested {len(results)} runs | REGRESSION class-flags={len(regr)} "
              f"across {len(regr_kernels)} kernels, {len(regr_shapes)} distinct failing shapes")
        for run, k, c in regr:
            print(f"    REGRESSION {run}  {c['class']} shape={c['shape']} "
                  f"a<->ref={c['a_vs_ref']:.4g}")
        summary["divergence"] = {
            "ran": True, "runs_tested": len(results),
            "regression_class_flags": len(regr),
            "regression_kernels": regr_kernels,
            "regression_distinct_shapes": [{"kernel": k, "shape": list(s)}
                                           for k, s in regr_shapes],
        }
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\n[wrote artifacts under {HERE}]")


if __name__ == "__main__":
    main()
