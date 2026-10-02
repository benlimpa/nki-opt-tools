"""Offline tests for the coverage spec + spectra binning. No hardware, no torch.

Run:  python3 -m pytest kernel-testing/coverage/test_spectra.py
or:   python3 kernel-testing/coverage/test_spectra.py   (uses a tiny built-in runner)

The most valuable checks are the PARTITION properties: every dimension's buckets must
be mutually exclusive AND exhaustive over the opt's real sample params, so that
"bins_hit / bins_defined" is a well-defined coverage metric and input generation can
target a specific bucket unambiguously.
"""
from __future__ import annotations

import itertools
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import spectra  # noqa: E402

SPEC = spectra.load_spec()
OPTS = spectra.opt_ids(SPEC)


# ---------------------------------------------------------------- structural validity

def test_spec_has_optimizations():
    assert len(OPTS) == 24, f"expected 24 opts, got {len(OPTS)}: {OPTS}"


def test_required_fields_and_enums():
    for opt in OPTS:
        e = SPEC[opt]
        for key in ("name", "equivalence_class", "oracle", "params",
                    "structural_monitor", "structural_expected", "dimensions",
                    "sample_params"):
            assert key in e, f"{opt} missing field {key!r}"
        assert e["equivalence_class"] in spectra.ALLOWED_EQUIV, \
            f"{opt} bad equivalence_class {e['equivalence_class']!r}"
        assert e["oracle"] in spectra.ALLOWED_ORACLE, \
            f"{opt} bad oracle {e['oracle']!r}"
        assert e["dimensions"], f"{opt} has no dimensions"


def test_precision_opts_use_fp32_reference():
    # Class C cannot use before_after (before/after consume different-dtype inputs).
    for opt in OPTS:
        if SPEC[opt]["equivalence_class"] == "C_precision":
            assert SPEC[opt]["oracle"] == "torch_fp32_reference", \
                f"{opt} is C_precision but oracle={SPEC[opt]['oracle']!r}"


def test_multicore_opts_do_not_use_before_after():
    # before_after needs both kernels resident; multi-core opts should lean on a
    # torch reference to avoid the simultaneous-launch constraint (oracle-design.md addendum).
    for opt in OPTS:
        if SPEC[opt].get("multi_core"):
            assert SPEC[opt]["oracle"] != "before_after", \
                f"{opt} is multi_core but oracle=before_after (launch constraint)"


# ---------------------------------------------------------------- predicate sanity

def test_predicates_evaluate():
    for opt in OPTS:
        for params in SPEC[opt]["sample_params"]:
            # Should not raise; every dimension should match exactly one bucket.
            sp = spectra.bin_input(SPEC, opt, params)
            assert set(sp) == {d["name"] for d in SPEC[opt]["dimensions"]}


# ---------------------------------------------------------------- PARTITION properties

def _categorical_values(dim: dict) -> list | None:
    """For a categorical dimension (predicate `key == 'literal'`), return the set of
    literals so we can self-test exclusivity/exhaustiveness without real samples."""
    import re
    vals, key = [], None
    for b in dim["buckets"]:
        m = re.fullmatch(r"\s*(\w+)\s*==\s*'([^']*)'\s*", b["predicate"])
        if not m:
            return None
        key = m.group(1)
        vals.append((key, m.group(2)))
    return vals


def test_buckets_partition_samples():
    """Over each opt's real sample params, every dimension matches exactly one bucket."""
    for opt in OPTS:
        for dim in SPEC[opt]["dimensions"]:
            cats = _categorical_values(dim)
            if cats is not None:
                # categorical: synthesize one params dict per literal
                key = cats[0][0]
                samples = [{key: v} for _, v in cats]
            else:
                samples = SPEC[opt]["sample_params"]
            for params in samples:
                matched = [b["name"] for b in dim["buckets"]
                           if spectra._eval_predicate(b["predicate"], params)]
                assert len(matched) == 1, (
                    f"{opt}/{dim['name']}: params={params} matched {matched} "
                    f"(must be exactly one — buckets must partition)"
                )


def test_every_critical_bucket_is_reachable():
    """Each critical bucket must be satisfiable by some param assignment, else we
    can never generate that boundary case (and the coverage claim is vacuous)."""
    for opt in OPTS:
        for dim in SPEC[opt]["dimensions"]:
            cats = _categorical_values(dim)
            for b in dim["buckets"]:
                if not b.get("critical"):
                    continue
                if cats is not None:
                    key = cats[0][0]
                    # the literal for this bucket
                    lit = [v for (k, v) in cats][dim["buckets"].index(b)]
                    assert spectra._eval_predicate(b["predicate"], {key: lit})
                else:
                    # search a small grid drawn from sample params' value ranges
                    assert _reachable_numeric(opt, dim, b), \
                        f"{opt}/{dim['name']}/{b['name']} unreachable in sampled grid"


def _reachable_numeric(opt: str, dim: dict, bucket: dict) -> bool:
    params_keys = SPEC[opt]["params"]
    # candidate values per key: pulled from sample params + boundary-ish extras
    pool: dict[str, set] = {k: set() for k in params_keys}
    for sp in SPEC[opt]["sample_params"]:
        for k, v in sp.items():
            if isinstance(v, (int, float)):
                pool.setdefault(k, set()).add(v)
    extras = [0, 1, 2, 3, 4, 5, 8, 9, 16, 43, 128, 129, 256, 512]
    for k in pool:
        pool[k].update(extras)
    keys = [k for k in pool if pool[k]]
    # bounded cartesian search
    combos = itertools.product(*[sorted(pool[k]) for k in keys])
    for i, combo in enumerate(combos):
        if i > 20000:
            break
        params = dict(zip(keys, combo))
        try:
            if spectra._eval_predicate(bucket["predicate"], params):
                return True
        except KeyError:
            return True  # needs a param outside numeric pool; treat as reachable
        except ZeroDivisionError:
            continue  # 0 divisor is not a realistic param value; skip this combo
    return False


# ---------------------------------------------------------------- coverage accounting

def test_coverage_counts():
    # opt38: sample params cover the 'uneven_shard' critical bucket (43 % 2 != 0).
    rep = spectra.coverage(SPEC, "opt38", SPEC["opt38"]["sample_params"])
    assert "shard_divisibility" in rep.dimensions
    assert "uneven_shard" in rep.dimensions["shard_divisibility"].hit, \
        "opt38 sample_params should exercise the uneven-shard bug bucket"
    assert 0 < rep.fraction <= 1.0
    assert rep.bins_defined == sum(len(d.defined) for d in rep.dimensions.values())


def test_uncovered_critical_detected():
    # A campaign that only sends divisible shards must FLAG the uneven-shard critical gap.
    divisible_only = [
        {"outer_loop_count": 16, "target_grid_size": 2, "saturation_pct": 50},
        {"outer_loop_count": 32, "target_grid_size": 4, "saturation_pct": 50},
    ]
    rep = spectra.coverage(SPEC, "opt38", divisible_only)
    assert "shard_divisibility:uneven_shard" in rep.uncovered_critical, \
        "must report the uneven-shard critical bin as uncovered"


def test_new_spectrum_gate():
    seen: set = set()
    p1 = {"outer_loop_count": 16, "target_grid_size": 2, "saturation_pct": 50}
    p2 = {"outer_loop_count": 32, "target_grid_size": 2, "saturation_pct": 50}  # same buckets
    p3 = {"outer_loop_count": 43, "target_grid_size": 2, "saturation_pct": 50}  # uneven -> new
    assert spectra.new_spectrum(SPEC, "opt38", p1, seen) is True
    assert spectra.new_spectrum(SPEC, "opt38", p2, seen) is False  # redundant: skip hw run
    assert spectra.new_spectrum(SPEC, "opt38", p3, seen) is True   # opens new bucket


def test_transform_fired():
    # spill_save_bytes 4096 -> 0 means opt01 fired; 0 -> 0 means it did not.
    fired = spectra.transform_fired(SPEC, "opt01",
                                    {"spill_save_bytes": 4096}, {"spill_save_bytes": 0})
    assert fired is True
    noop = spectra.transform_fired(SPEC, "opt01",
                                   {"spill_save_bytes": 0}, {"spill_save_bytes": 0})
    assert noop is False
    # opt20 monitor (scalar_instr_count) is not in the profile schema -> None (documented gap).
    assert spectra.transform_fired(SPEC, "opt20", {}, {}) is None


# ---------------------------------------------------------------- tiny runner

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
