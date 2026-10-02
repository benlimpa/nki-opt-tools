"""Offline tests for the coverage-guided generator. No hardware, no torch.

Run:  python3 kernel-testing/coverage/test_generate.py

The key property: for every optimization, the generated input set must cover ALL
reachable bins — and in particular every CRITICAL bin (the boundary/negative cases the
existing fixed-shape calibration sweeps miss). If this regresses, the campaign would
silently stop exercising the bug-revealing regimes.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import generate  # noqa: E402
import spectra  # noqa: E402

SPEC = spectra.load_spec()
OPTS = spectra.opt_ids(SPEC)


def test_every_opt_generates_inputs():
    for opt in OPTS:
        inputs = generate.generate(SPEC, opt)
        assert inputs, f"{opt}: generator produced no inputs"


def test_generated_inputs_bin_cleanly():
    for opt in OPTS:
        for params in generate.generate(SPEC, opt):
            sp = spectra.bin_input(SPEC, opt, params)  # raises if it doesn't bin
            assert set(sp) == {d["name"] for d in SPEC[opt]["dimensions"]}


def test_full_coverage_of_reachable_bins():
    """Generated inputs cover the entire reachable bin space for every opt."""
    for opt in OPTS:
        inputs = generate.generate(SPEC, opt)
        # reachable bins = those any candidate can hit
        reachable = set()
        for params, pairs in (
            (p, generate._spectrum_pairs(SPEC, opt, p)) for p in _enumerate_pool(opt)
        ):
            if pairs:
                reachable |= pairs
        covered = set()
        for params in inputs:
            covered |= set(spectra.bin_input(SPEC, opt, params).items())
        missing = reachable - covered
        assert not missing, f"{opt}: generator left bins uncovered: {sorted(missing)}"


def test_all_critical_bins_covered():
    """Every critical bin is hit — the boundary/negative regimes are the point."""
    for opt in OPTS:
        inputs = generate.generate(SPEC, opt)
        rep = spectra.coverage(SPEC, opt, inputs)
        assert not rep.uncovered_critical, \
            f"{opt}: uncovered critical bins after generation: {rep.uncovered_critical}"


def test_opt38_generates_uneven_shard():
    # The motivating bug regime must appear concretely.
    inputs = generate.generate(SPEC, "opt38")
    assert any(p["outer_loop_count"] % p["target_grid_size"] != 0 for p in inputs), \
        "opt38 generator must produce an uneven-shard input (the half-output bug regime)"


def test_determinism():
    for opt in ("opt18", "opt38", "opt13"):
        assert generate.generate(SPEC, opt) == generate.generate(SPEC, opt), \
            f"{opt}: generation is not deterministic"


def test_generated_shapes_are_executable():
    """Guard against degenerate, non-runnable shapes (e.g. K=0, M=1 matmul). No
    numeric value is 0, and a param whose real samples are large (>=64, i.e. a
    matmul/free dimension) is never synthesized below 16."""
    for opt in OPTS:
        sample_min = {}
        for sp in SPEC[opt]["sample_params"]:
            for k, v in sp.items():
                if isinstance(v, int) and not isinstance(v, bool):
                    sample_min[k] = min(sample_min.get(k, v), v)
        for params in generate.generate(SPEC, opt):
            for k, v in params.items():
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    continue
                assert v != 0, f"{opt}: generated degenerate {k}=0 in {params}"
                if sample_min.get(k, 0) >= 64:
                    assert v >= 16, f"{opt}: {k}={v} far below real range in {params}"


# ---------------------------------------------------------------- helpers

def _enumerate_pool(opt: str):
    import itertools
    pool = generate.candidate_pool(SPEC, opt)
    keys = list(pool)
    for combo in itertools.product(*[pool[k] for k in keys]):
        yield dict(zip(keys, combo))


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
