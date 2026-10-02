"""Independent correctness sweep by optimization METHOD (approach).

For every run, run BOTH the baseline and the optimized kernel through the run's
own `oracle.run_test` (the kernel's declared correctness test), and record the
outcome. This is independent of the agent's self-reported success.

The method-attributable signal is a REGRESSION: baseline passes run_test but the
optimized kernel fails it -> the method broke a correct kernel. (A run where the
baseline already fails is not the method's fault and is excluded from the
regression rate.)

Writes a per-run CSV and prints a by-approach summary. Needs the Neuron venv +
device. Usage:
    method_correctness.py [--skip mamba,...] [--out path.csv] [--limit N]
"""
from __future__ import annotations

import csv
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import run_index as ri  # noqa: E402

_REPO = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))


def _run_test_outcome(kernel_path: pathlib.Path, oracle_path: pathlib.Path, tag: str) -> str:
    """PASS / FAIL / ERROR / NORUNTEST for one kernel file vs its oracle.run_test."""
    import kernel_oracle as ko  # noqa: PLC0415
    from triton_nki_transpile.utils import extract_wrapper_name  # noqa: PLC0415
    try:
        kmod = ko._load_module(kernel_path, f"_mc_{tag}")
        omod = ko._load_module(oracle_path, f"_mco_{tag}")
        run_test = getattr(omod, "run_test", None)
        if run_test is None:
            return "NORUNTEST"
        w = getattr(kmod, extract_wrapper_name(kernel_path.read_text()))
    except Exception:  # noqa: BLE001
        return "ERROR"
    try:
        run_test(w)
        return "PASS"
    except AssertionError:
        return "FAIL"
    except Exception:  # noqa: BLE001
        return "ERROR"


def _balance(runs, per_approach):
    """Up to `per_approach` runs per approach, maximizing kernel diversity
    (round-robin over kernels within each approach)."""
    import collections  # noqa: PLC0415
    by_appr = collections.defaultdict(lambda: collections.defaultdict(list))
    for p in runs:
        by_appr[p.approach][p.kernel].append(p)
    out = []
    for appr, kernels in by_appr.items():
        picked, lists = [], [list(v) for v in kernels.values()]
        i = 0
        while len(picked) < per_approach and any(lists):
            lst = lists[i % len(lists)]
            if lst:
                picked.append(lst.pop())
            if not any(lists):
                break
            i += 1
        out.extend(picked)
    return out


def sweep(skip_kernels=frozenset(), out_path: pathlib.Path | None = None,
          limit=None, per_approach=None):
    import collections  # noqa: PLC0415
    # manifest: run_dir name -> agent_features (v1_opt37 / v2_opt38 ...)
    feats = {}
    man = _REPO / "benchmark-opt" / "manifest.csv"
    if man.exists():
        for r in csv.DictReader(open(man)):
            feats[r["run_dir"].split("/")[-1]] = r.get("agent_features", "")

    rep = ri.index_runs()
    runs = [p for p in rep.pairs if p.has_pair and p.kernel not in skip_kernels]
    if per_approach:
        runs = _balance(runs, per_approach)
    if limit:
        runs = runs[:limit]

    rows = []
    for i, p in enumerate(runs):
        base = _run_test_outcome(p.baseline_path, p.oracle_path, f"b{i}")
        opt = _run_test_outcome(p.optimized_path, p.oracle_path, f"o{i}")
        af = feats.get(p.run_dir.name, "")
        rows.append({"approach": p.approach, "agent_features": af,
                     "source": p.source, "kernel": p.kernel,
                     "baseline": base, "optimized": opt,
                     "regression": "YES" if (base == "PASS" and opt != "PASS") else "",
                     "run": p.run_dir.name})
        print(f"[{i+1}/{len(runs)}] {p.approach:15s} {p.source}/{p.kernel:20s} "
              f"base={base:4s} opt={opt:4s}"
              f"{'  <-- REGRESSION' if rows[-1]['regression'] else ''}", flush=True)

    if out_path:
        with open(out_path, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader(); wr.writerows(rows)
        print(f"\n[wrote {out_path}]")

    # summary by approach
    print("\n===== by approach =====")
    print(f"{'approach':16s} {'runs':>4s} {'base_ok':>7s} {'opt_ok':>6s} "
          f"{'regress(opt fail|base ok)':>26s}")
    agg = collections.defaultdict(lambda: collections.Counter())
    for r in rows:
        a = agg[r["approach"]]
        a["n"] += 1
        a["base_ok"] += r["baseline"] == "PASS"
        a["opt_ok"] += r["optimized"] == "PASS"
        a["base_ok_runs"] += r["baseline"] == "PASS"
        a["regress"] += r["regression"] == "YES"
    for ap in sorted(agg):
        a = agg[ap]
        denom = a["base_ok_runs"] or 1
        print(f"{ap:16s} {a['n']:>4d} {a['base_ok']:>7d} {a['opt_ok']:>6d} "
              f"{a['regress']:>5d}/{a['base_ok_runs']:<3d} ({100*a['regress']/denom:.0f}%)")
    return rows


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", default="mamba")
    ap.add_argument("--out", default="kernel-testing/coverage/method_correctness.csv")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--per-approach", type=int, default=0)
    args = ap.parse_args()
    sweep(skip_kernels=frozenset(k for k in args.skip.split(",") if k),
          out_path=pathlib.Path(args.out) if args.out else None,
          limit=args.limit or None, per_approach=args.per_approach or None)
