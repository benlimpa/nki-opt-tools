"""Coverage-guided input generator (offline, deterministic).

Given the coverage spec, synthesize a small set of concrete param dicts per optimization
that covers every *reachable* regime bin — biased toward the critical boundary/negative
bins the existing calibration sweeps miss (e.g. uneven shards, non-divisible tiles,
near-overflow dtypes).

This is the cheap "guide" layer of the HeteroFuzz adaptation (see ../heterofuzz-mechanism.md):
it proposes inputs in regime-space without touching hardware. The campaign then runs each
proposed input through the oracle, gating expensive device runs with spectra.new_spectrum.

Pure Python, no torch, no device:
  - candidate_pool(spec, opt)        -> {param: [values]}   (seeds + predicate-derived boundaries)
  - generate(spec, opt, budget)      -> [params]            (greedy set-cover of the bins)
  - generate_all(spec, budget)       -> {opt: [params]}
"""
from __future__ import annotations

import itertools
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import spectra  # noqa: E402

# Default per-combination enumeration cap so a wide param space can't blow up.
_MAX_CANDIDATES = 50_000


# ---------------------------------------------------------------- candidate pools

_CAT_RE = re.compile(r"(\w+)\s*==\s*'([^']*)'")
_BOOL_RE = re.compile(r"(\w+)\s*==\s*(True|False)")
_INT_RE = re.compile(r"\b(\d+)\b")
_WORD_RE = re.compile(r"[A-Za-z_]\w*")


def _predicate_literals(entry: dict) -> tuple[dict[str, set], dict[str, set], dict[str, set]]:
    """Scan an opt's bucket predicates for, per param: categorical literals, bool
    literals, and integer thresholds *attributed to the params named in the same
    predicate*. Per-param attribution (not a global int pool) keeps a tile threshold
    like 128 out of a matmul-dimension param's candidate values, so we don't synthesize
    nonsense shapes like K=0 just because some other predicate mentions 0."""
    cats: dict[str, set] = {}
    bools: dict[str, set] = {}
    param_ints: dict[str, set] = {}
    for dim in entry["dimensions"]:
        for b in dim["buckets"]:
            pred = b["predicate"]
            for name, lit in _CAT_RE.findall(pred):
                cats.setdefault(name, set()).add(lit)
            for name, lit in _BOOL_RE.findall(pred):
                bools.setdefault(name, set()).add(lit == "True")
            names = {w for w in _WORD_RE.findall(pred) if w not in {"and", "or", "not"}}
            ints = {int(t) for t in _INT_RE.findall(pred)}
            for nm in names:
                param_ints.setdefault(nm, set()).update(ints)
    return cats, bools, param_ints


def candidate_pool(spec: dict, opt: str) -> dict[str, list]:
    """Per-param candidate values: categorical/bool literals as-is; numeric params get
    real sample values + the thresholds attributed to *that* param + neighbors (n-1, n+1)
    so boundary bins (`>= 128`, `% TILE_K == 0`, `< 3`) are reachable with realistic,
    executable shapes. 0 is never emitted (no kernel dimension is 0)."""
    entry = spec[opt]
    cats, bools, param_ints = _predicate_literals(entry)

    sample_numeric: dict[str, set] = {}
    for sp in entry["sample_params"]:
        for k, v in sp.items():
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                sample_numeric.setdefault(k, set()).add(v)

    # Real integer sample values seen for ANY numeric param. Cross-pollinating these
    # lets cross-param predicates be satisfied (e.g. opt01 `M <= 2*TILE_M` needs an M
    # value as small as a real TILE_M) while every candidate stays an executable shape.
    cross_ints: set[int] = {
        v for vs in sample_numeric.values() for v in vs if isinstance(v, int)
    }

    pool: dict[str, list] = {}
    for p in entry["params"]:
        if p in cats:
            pool[p] = sorted(cats[p])
            continue
        if p in bools:
            pool[p] = sorted(bools[p] | {True, False})
            continue
        samples = sample_numeric.get(p, set())
        vals: set = set(samples) | cross_ints
        # +1 of each real sample breaks divisibility realistically (e.g. K=513 vs 512)
        vals |= {v + 1 for v in samples if isinstance(v, int)}
        # thresholds attributed to this param, with straddlers, to reach >=/< boundaries
        for n in param_ints.get(p, set()):
            vals.update({n - 1, n, n + 1})
        vals = {v for v in vals if (isinstance(v, float) and v > 0) or (isinstance(v, int) and v >= 1)}
        # Scale floor: for a large-scale param (real samples all >= 64, i.e. a matmul/
        # scan/free dimension), drop tiny threshold straddlers like 1 that come from
        # modulo/`==0` predicate constants — they'd be non-executable shapes.
        int_samples = {v for v in samples if isinstance(v, int)}
        if int_samples and min(int_samples) >= 64:
            vals = {v for v in vals if not (isinstance(v, int) and v < 16)}
        pool[p] = sorted(vals) or [1]
    return pool


def _realism(spec: dict, opt: str, params: dict) -> int:
    """How many of an input's values come straight from the real sweep samples.
    Used only as a tie-break so executable, sweep-shaped inputs win when two
    candidates cover the same new bins."""
    sample_numeric: dict[str, set] = {}
    for sp in spec[opt]["sample_params"]:
        for k, v in sp.items():
            sample_numeric.setdefault(k, set()).add(v)
    return sum(1 for k, v in params.items() if v in sample_numeric.get(k, set()))


# ---------------------------------------------------------------- generation

def _all_bins(entry: dict) -> tuple[set, set]:
    """Return (all_bins, critical_bins) as {(dim, bucket)} pairs."""
    allb, crit = set(), set()
    for dim in entry["dimensions"]:
        for b in dim["buckets"]:
            allb.add((dim["name"], b["name"]))
            if b.get("critical"):
                crit.add((dim["name"], b["name"]))
    return allb, crit


def _spectrum_pairs(spec: dict, opt: str, params: dict) -> set | None:
    """The bin pairs an input covers, or None if it doesn't bin cleanly."""
    try:
        return set(spectra.bin_input(spec, opt, params).items())
    except (ValueError, ZeroDivisionError):
        # ValueError: no/ambiguous bucket. ZeroDivisionError: a 0-divisor combo
        # (e.g. TILE_K_current=0) — not a realistic input; skip it.
        return None


def generate(spec: dict, opt: str, budget: int = 12,
             max_candidates: int = _MAX_CANDIDATES) -> list[dict]:
    """Greedy set-cover: pick inputs that open the most still-uncovered bins, breaking
    ties toward critical bins. Stops when all reachable bins are covered or budget hit."""
    entry = spec[opt]
    pool = candidate_pool(spec, opt)
    keys = list(pool)

    # enumerate valid candidates with their covered bin-sets (bounded)
    candidates: list[tuple[dict, set]] = []
    for i, combo in enumerate(itertools.product(*[pool[k] for k in keys])):
        if i >= max_candidates:
            break
        params = dict(zip(keys, combo))
        pairs = _spectrum_pairs(spec, opt, params)
        if pairs:
            candidates.append((params, pairs))

    reachable = set().union(*(p for _, p in candidates)) if candidates else set()
    _, critical = _all_bins(entry)

    chosen: list[dict] = []
    covered: set = set()
    while candidates and covered != reachable and len(chosen) < budget:
        def score(item):
            params, pairs = item
            new = pairs - covered
            # critical-new first, then total-new, then prefer realistic (sweep-shaped) inputs
            return (len(new & critical), len(new), _realism(spec, opt, params))
        params, pairs = max(candidates, key=score)
        if not (pairs - covered):
            break  # no remaining candidate adds anything
        chosen.append(params)
        covered |= pairs
        candidates = [(p, pr) for (p, pr) in candidates if p != params]
    return chosen


def generate_all(spec: dict, budget: int = 12) -> dict[str, list[dict]]:
    return {opt: generate(spec, opt, budget) for opt in spectra.opt_ids(spec)}


# ---------------------------------------------------------------- CLI

def _main() -> None:
    spec = spectra.load_spec()
    total_inputs = 0
    for opt in spectra.opt_ids(spec):
        inputs = generate(spec, opt)
        rep = spectra.coverage(spec, opt, inputs)
        total_inputs += len(inputs)
        flag = "" if not rep.uncovered_critical else \
            f"  !! still-uncovered critical: {', '.join(rep.uncovered_critical)}"
        print(f"{opt}: {len(inputs):2d} inputs -> {rep.bins_hit}/{rep.bins_defined} bins"
              f" ({rep.fraction*100:.0f}%){flag}")
    print(f"\ntotal generated inputs: {total_inputs}")


if __name__ == "__main__":
    _main()
