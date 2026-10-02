"""Offline tests for campaign's pure planning + coverage tally (no torch/device).

The device path (run_carrier/run_campaign -> kernel_oracle.run_pair) is exercised
on hardware, not here.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import campaign as cp  # noqa: E402
import grounding as gr  # noqa: E402

_ORACLE_MN = ("def _make_inputs(M, N, seed=42):\n    return (M, N)\n"
              "def _ref(a):\n    return a\n")
_KERNEL_N2048 = ("N_TILE = 2048\n"
                 "def k(x):\n    M, N = x.shape\n    t = N // N_TILE\n    return x\n")


def _grounding_N2048():
    return gr.ground_sources("nkibench", "adamw", _ORACLE_MN, _KERNEL_N2048, _KERNEL_N2048)


# ---- plan_shapes ------------------------------------------------------------

def test_plan_has_base_plus_one_per_critical():
    g = _grounding_N2048()                # 3 classes: exact, partial(crit), below(crit)
    plans = cp.plan_shapes(g)
    assert len(plans) == 3                # base + 2 criticals
    # base is all-exact: N = one full tile (max divisor)
    assert plans[0].assignment["N"] == 2048


def test_plan_violating_values():
    g = _grounding_N2048()
    plans = cp.plan_shapes(g)
    byclass = {p.targets[0]: p.assignment["N"] for p in plans[1:]}
    assert byclass["N_x2048_partial"] == 2049     # D+1
    assert byclass["N_x2048_below"] == 1024       # D//2


def test_plan_empty_when_ungrounded():
    g = gr.ground_sources("x", "k", "def _ref(): pass", "def k(): pass", "def k(): pass")
    assert cp.plan_shapes(g) == []


# ---- coverage ---------------------------------------------------------------

def test_coverage_full_from_planned_shapes():
    g = _grounding_N2048()
    plans = cp.plan_shapes(g)
    cov = cp.coverage(g, [p.assignment for p in plans])
    assert cov.bins_defined == 3
    assert cov.bins_hit == 3
    assert cov.uncovered == []
    assert cov.uncovered_critical == []
    assert abs(cov.fraction - 1.0) < 1e-9


def test_coverage_surfaces_uncovered_critical():
    g = _grounding_N2048()
    # only an exact shape -> the two critical bins are uncovered
    cov = cp.coverage(g, [{"M": 256, "N": 4096}])
    assert cov.bins_hit == 1
    assert set(cov.uncovered_critical) == {"N_x2048_partial", "N_x2048_below"}


def test_coverage_partition_is_exclusive():
    # each planned shape's targeted critical class is the only critical it hits
    g = _grounding_N2048()
    plans = cp.plan_shapes(g)
    for p in plans[1:]:
        cov = cp.coverage(g, [p.assignment])
        crit_hit = [n for n in cov.class_hits if n.endswith(("partial", "below"))]
        assert crit_hit == p.targets, f"{p.targets} -> {crit_hit}"


def test_multi_knob_base_covers_all_exact():
    # matmul-like: M,K,N each tiled -> base must satisfy all three *_exact
    g = gr.ground_sources(
        "nkibench", "matmul", "def _make_inputs(M, K, N, seed=42):\n    return (M,K,N)\n",
        "P=128\nN_TILE=512\ndef k(a,b):\n    K,M=a.shape\n    _,N=b.shape\n"
        "    x=M//P\n    y=K//P\n    z=N//N_TILE\n    return a\n",
        "P=128\nN_TILE=512\ndef k(a,b):\n    K,M=a.shape\n    _,N=b.shape\n"
        "    x=M//P\n    y=K//P\n    z=N//N_TILE\n    return a\n")
    base = cp.plan_shapes(g)[0]
    exact = cp.coverage(g, [base.assignment])
    assert {n for n in exact.class_hits if n.endswith("exact")} == {
        "M_x128_exact", "K_x128_exact", "N_x512_exact"}


def _run() -> int:
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns)-failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run())
