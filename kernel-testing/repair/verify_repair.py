"""Verify a repaired NKI kernel: (1) the flagged divergence is gone, (2) the
speedup is maintained.

Given the three kernels in a repair run — the correct-but-slow baseline, the
fast-but-wrong "broken" optimized kernel, and the agent's repaired kernel — plus
the shared oracle, this:

  1. Re-runs the coverage-guided differential oracle (the same engine as
     kernel-testing/coverage/cell_divergence.py) on
       BEFORE = baseline vs broken     (should reproduce the divergence)
       AFTER  = baseline vs repaired   (should show no REGRESSION)
     A repair is correct iff AFTER has zero REGRESSION classes (and the
     specifically-flagged class, if given, is no longer a REGRESSION).

  2. Profiles broken and repaired on hardware (and the baseline) and reports
     whether the repaired kernel is still faster than the baseline and how much
     of the broken kernel's speedup it retained.

Writes verdict.json and prints a summary. Needs the Neuron venv + a trn1 device;
run with NEURON_PLATFORM_TARGET_OVERRIDE=trn1.

Usage:
    verify_repair.py --baseline kernel.py --broken broken.py --repaired optimized_kernel.py \
        --oracle oracle.py --source reference --kernel rms --out verdict.json \
        [--flagged-class rows_x128_partial] [--baseline-latency-ms 1.234]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import subprocess
import sys

_HERE = pathlib.Path(__file__).resolve().parent
_COV = _HERE.parent / "coverage"
sys.path.insert(0, str(_COV))
_REPO = _HERE.parents[1]


@dataclasses.dataclass
class _Carrier:
    """Minimal stand-in for run_index.RunPair (what the divergence engine needs)."""
    baseline_path: pathlib.Path
    optimized_path: pathlib.Path
    oracle_path: pathlib.Path
    source: str
    kernel: str


def _divergence(baseline, optimized, oracle, source, kernel):
    """Run the differential oracle in a SUBPROCESS. Running it in-process leaves the
    NRT/XLA runtime holding both neuron cores for the life of this process, which
    then starves the profiler subprocesses ('Logical Neuron Core(s) not available -
    Requested:2 Available:0'). A subprocess fully releases the device on exit."""
    cmd = [sys.executable, str(_HERE / "verify_repair.py"), "--diff-worker",
           str(baseline), str(optimized), str(oracle), source, kernel]
    p = subprocess.run(cmd, capture_output=True, text=True,
                       env=_os_environ_with_pp(f"{_REPO/'nki-library'/'src'}:{_REPO/'src'}"))
    for line in reversed(p.stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return {"classes": [], "error": (p.stderr or p.stdout or "diff worker: no output")[-800:]}


def _profile(kernel_path, oracle_path):
    """Latency (ms) of one kernel via profile_cli (subprocess for NKI isolation)."""
    env_pp = f"{_REPO/'nki-library'/'src'}:{_REPO/'src'}"
    cmd = [sys.executable, "-m", "triton_nki_transpile.profile_cli",
           "--kernel-path", str(kernel_path), "--oracle-path", str(oracle_path), "--bench"]
    p = subprocess.run(cmd, capture_output=True, text=True,
                       env={**_os_environ_with_pp(env_pp)})
    out = p.stdout.strip().splitlines()
    for line in reversed(out):                       # last JSON line is the result
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return {"success": False, "latency_ms": None,
            "error": (p.stderr or p.stdout or "no profiler output")[-500:]}


def _os_environ_with_pp(pp):
    import os  # noqa: PLC0415
    e = dict(os.environ)
    e["PYTHONPATH"] = pp + (":" + e["PYTHONPATH"] if e.get("PYTHONPATH") else "")
    return e


def _regressions(div):
    return [c for c in div.get("classes", [])
            if c.get("ran", True) and c.get("grade") == "REGRESSION"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="correct-but-slow kernel (kernel.py)")
    ap.add_argument("--broken", required=True, help="fast-but-wrong optimized kernel")
    ap.add_argument("--repaired", required=True, help="agent's repaired optimized_kernel.py")
    ap.add_argument("--oracle", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--kernel", required=True)
    ap.add_argument("--flagged-class", default=None,
                    help="the specific behavior class that was flagged (optional)")
    ap.add_argument("--baseline-latency-ms", type=float, default=None,
                    help="cached baseline latency; if omitted the baseline is profiled")
    ap.add_argument("--out", default="verdict.json")
    ap.add_argument("--speedup-retention", type=float, default=0.90,
                    help="fraction of the broken kernel's speedup the repair must keep")
    args = ap.parse_args()

    print("== 1. divergence: BEFORE (baseline vs broken) ==", flush=True)
    before = _divergence(args.baseline, args.broken, args.oracle, args.source, args.kernel)
    reg_before = _regressions(before)
    print(f"   baseline_run_test={before.get('baseline_passes_run_test')}  "
          f"REGRESSIONS={len(reg_before)}")
    for c in reg_before:
        print(f"     {c['class']:20s} {c['shape']}  {c['detail']}")

    print("== 1. divergence: AFTER (baseline vs repaired) ==", flush=True)
    after = _divergence(args.baseline, args.repaired, args.oracle, args.source, args.kernel)
    reg_after = _regressions(after)
    print(f"   REGRESSIONS={len(reg_after)}")
    for c in reg_after:
        print(f"     {c['class']:20s} {c['shape']}  {c['detail']}")

    flagged_fixed = None
    if args.flagged_class:
        still = any(c["class"] == args.flagged_class for c in reg_after)
        flagged_fixed = not still
    divergence_fixed = len(reg_after) == 0

    print("== 2. speedup ==", flush=True)
    prof_broken = _profile(args.broken, args.oracle)
    prof_repaired = _profile(args.repaired, args.oracle)
    lat_broken = prof_broken.get("latency_ms")
    lat_repaired = prof_repaired.get("latency_ms")
    lat_baseline = args.baseline_latency_ms
    if lat_baseline is None:
        prof_baseline = _profile(args.baseline, args.oracle)
        lat_baseline = prof_baseline.get("latency_ms")

    def _sp(num, den):
        return (num / den) if (num and den) else None
    speedup_broken = _sp(lat_baseline, lat_broken)
    speedup_repaired = _sp(lat_baseline, lat_repaired)
    # fraction of the broken kernel's speed the repair retained (1.0 = identical)
    retention = _sp(lat_broken, lat_repaired)
    repaired_runs = bool(prof_repaired.get("success")) and lat_repaired is not None
    still_faster_than_baseline = bool(speedup_repaired and speedup_repaired > 1.0)
    speedup_maintained = bool(
        repaired_runs and still_faster_than_baseline
        and (retention is not None and retention >= args.speedup_retention))

    print(f"   baseline={lat_baseline} ms  broken={lat_broken} ms  repaired={lat_repaired} ms")
    print(f"   speedup broken={speedup_broken}  repaired={speedup_repaired}  "
          f"retention={retention}")

    overall = bool(divergence_fixed and repaired_runs and speedup_maintained)
    verdict = {
        "source": args.source, "kernel": args.kernel,
        "flagged_class": args.flagged_class, "flagged_class_fixed": flagged_fixed,
        "divergence": {
            "regressions_before": len(reg_before),
            "regressions_after": len(reg_after),
            "divergence_fixed": divergence_fixed,
            "before_classes": before.get("classes", []),
            "after_classes": after.get("classes", []),
        },
        "speedup": {
            "latency_ms": {"baseline": lat_baseline, "broken": lat_broken,
                           "repaired": lat_repaired},
            "speedup_vs_baseline": {"broken": speedup_broken, "repaired": speedup_repaired},
            "retention_vs_broken": retention,
            "repaired_runs": repaired_runs,
            "still_faster_than_baseline": still_faster_than_baseline,
            "speedup_maintained": speedup_maintained,
            "profile_broken": prof_broken, "profile_repaired": prof_repaired,
        },
        "overall_success": overall,
    }
    pathlib.Path(args.out).write_text(json.dumps(verdict, indent=2) + "\n")

    print("\n== VERDICT ==")
    print(f"   divergence_fixed   : {divergence_fixed} "
          f"({len(reg_before)} -> {len(reg_after)} regressions)"
          + (f"  flagged[{args.flagged_class}]_fixed={flagged_fixed}" if args.flagged_class else ""))
    print(f"   speedup_maintained : {speedup_maintained} "
          f"(repaired {speedup_repaired and round(speedup_repaired,3)}x vs baseline, "
          f"kept {retention and round(retention*100)}% of broken speed)")
    print(f"   OVERALL SUCCESS    : {overall}")
    print(f"   [wrote {args.out}]")
    return 0 if overall else 1


def _diff_worker():
    """Subprocess entry: run the differential oracle once, print the result JSON as
    the last stdout line. Isolated so the device is released before profiling."""
    import cell_divergence as cd  # noqa: PLC0415
    baseline, optimized, oracle, source, kernel = sys.argv[2:7]
    carrier = _Carrier(pathlib.Path(baseline), pathlib.Path(optimized),
                       pathlib.Path(oracle), source, kernel)
    print(json.dumps(cd.analyze(carrier)))


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--diff-worker":
        _diff_worker()
    else:
        sys.exit(main())
