"""Optimization-behavioral coverage (kernel spectra) — offline core.

Adapts HeteroFuzz value-spectra monitoring (see ../heterofuzz-mechanism.md) to NKI
optimizations. The "spectrum" of an input is which *regime bucket* it lands in, per
optimization, along each coverage dimension defined in coverage_spec.toml. Coverage of
that bucket space is the empirical adequacy metric; the differential/torch oracle
(separate, see ../oracle-design.md) is what turns a covered bucket into a checked one.

This module is pure and hardware-free:
  - load_spec(path)                      -> dict
  - bin_input(spec, opt, params)         -> {dimension: bucket}     (the input spectrum)
  - new_spectrum(spec, opt, params, seen)-> bool                    (HeteroFuzz novelty gate)
  - coverage(spec, opt, samples)         -> CoverageReport          (bins_hit / bins_defined)
  - transform_fired(spec, opt, before, after) -> bool|None          (structural firing check)

The profile-counter side (transform_fired) only *interprets* an already-collected
neuron-profile dict; it does not run hardware. Campaign/oracle execution lives elsewhere.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass, field

try:
    import tomllib  # py3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

# ---------------------------------------------------------------- spec loading

_SPEC_DEFAULT = pathlib.Path(__file__).with_name("coverage_spec.toml")

# keys that are top-level scalars, not opt entries
_NON_OPT_KEYS = {"schema_version"}

ALLOWED_EQUIV = {"A_bitexact", "B_fp_reorder", "C_precision"}
ALLOWED_ORACLE = {"torch_reference", "before_after", "torch_fp32_reference"}


def load_spec(path: str | pathlib.Path = _SPEC_DEFAULT) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def opt_ids(spec: dict) -> list[str]:
    return [k for k in spec if k not in _NON_OPT_KEYS]


# ---------------------------------------------------------------- safe predicate eval

# Restricted environment: arithmetic + comparison over params only. No builtins,
# no attribute access. True/False/None are provided because predicates compare bool
# knobs (e.g. has_private_hbm_intermediate == True).
_SAFE_GLOBALS = {"__builtins__": {}, "True": True, "False": False, "None": None,
                 "min": min, "max": max, "abs": abs}


def _eval_predicate(predicate: str, params: dict) -> bool:
    try:
        return bool(eval(predicate, dict(_SAFE_GLOBALS), dict(params)))  # noqa: S307
    except NameError as e:
        raise KeyError(
            f"predicate {predicate!r} references a param not provided: {e}"
        ) from e


# ---------------------------------------------------------------- binning

def bucket_of(dimension: dict, params: dict) -> str | None:
    """Return the name of the (single) matching bucket, or None if no bucket matches."""
    hits = [b["name"] for b in dimension["buckets"] if _eval_predicate(b["predicate"], params)]
    if len(hits) > 1:
        raise ValueError(
            f"dimension {dimension['name']!r}: params matched >1 bucket {hits} "
            f"(buckets must partition the space) for params={params}"
        )
    return hits[0] if hits else None


def bin_input(spec: dict, opt: str, params: dict) -> dict[str, str]:
    """The input spectrum: one bucket per dimension. Raises if any dimension is unmatched."""
    entry = spec[opt]
    out: dict[str, str] = {}
    for dim in entry["dimensions"]:
        b = bucket_of(dim, params)
        if b is None:
            raise ValueError(
                f"{opt} dimension {dim['name']!r}: no bucket matched params={params} "
                f"(buckets must be exhaustive)"
            )
        out[dim["name"]] = b
    return out


def spectrum_key(spec: dict, opt: str, params: dict) -> frozenset[tuple[str, str]]:
    return frozenset(bin_input(spec, opt, params).items())


def new_spectrum(spec: dict, opt: str, params: dict,
                 seen: set[frozenset]) -> bool:
    """HeteroFuzz novelty gate: True (and records) iff this input opens an unseen bucket combo.

    The campaign uses this to decide whether an input is worth an expensive Trainium
    run — mirroring HeteroFuzz's selective HLS-simulator invocation.
    """
    k = spectrum_key(spec, opt, params)
    if k in seen:
        return False
    seen.add(k)
    return True


# ---------------------------------------------------------------- coverage report

@dataclass
class DimensionCoverage:
    name: str
    hit: set[str] = field(default_factory=set)
    defined: list[str] = field(default_factory=list)
    critical: set[str] = field(default_factory=set)

    @property
    def uncovered(self) -> list[str]:
        return [b for b in self.defined if b not in self.hit]

    @property
    def uncovered_critical(self) -> list[str]:
        return [b for b in self.defined if b in self.critical and b not in self.hit]


@dataclass
class CoverageReport:
    opt: str
    dimensions: dict[str, DimensionCoverage]

    @property
    def bins_defined(self) -> int:
        return sum(len(d.defined) for d in self.dimensions.values())

    @property
    def bins_hit(self) -> int:
        return sum(len(d.hit) for d in self.dimensions.values())

    @property
    def fraction(self) -> float:
        return self.bins_hit / self.bins_defined if self.bins_defined else 1.0

    @property
    def uncovered_critical(self) -> list[str]:
        return [f"{d.name}:{b}" for d in self.dimensions.values()
                for b in d.uncovered_critical]

    def summary(self) -> str:
        crit = self.uncovered_critical
        line = (f"{self.opt}: {self.bins_hit}/{self.bins_defined} bins "
                f"({self.fraction*100:.0f}%)")
        if crit:
            line += f"  !! UNCOVERED CRITICAL: {', '.join(crit)}"
        return line


def _dimension_skeleton(entry: dict) -> dict[str, DimensionCoverage]:
    dims: dict[str, DimensionCoverage] = {}
    for dim in entry["dimensions"]:
        dc = DimensionCoverage(name=dim["name"])
        for b in dim["buckets"]:
            dc.defined.append(b["name"])
            if b.get("critical"):
                dc.critical.add(b["name"])
        dims[dim["name"]] = dc
    return dims


def coverage(spec: dict, opt: str, samples: list[dict]) -> CoverageReport:
    """Tally which buckets the given input samples have hit, per dimension."""
    entry = spec[opt]
    dims = _dimension_skeleton(entry)
    for params in samples:
        for dim in entry["dimensions"]:
            b = bucket_of(dim, params)
            if b is not None:
                dims[dim["name"]].hit.add(b)
    return CoverageReport(opt=opt, dimensions=dims)


# ---------------------------------------------------------------- profile (structural) side

# Counters present in nki-opt-models-v2/_common.py KernelProfile. structural_monitor
# fields not in this set (e.g. scalar_instr_count for opt20) are documented gaps:
# the profile schema would need extending to evaluate them on hardware.
PROFILE_COUNTERS = {
    "t_total_us", "t_active_us", "t_tensor_us", "t_vector_us", "t_scalar_us",
    "t_gpsimd_us", "t_dma_us", "t_static_dma_us", "hbm_read_bytes", "hbm_write_bytes",
    "dma_packet_count", "static_dma_packet_count", "vector_instr_count",
    "tensor_instr_count", "matmul_instr_count", "spill_save_bytes", "spill_reload_bytes",
}


def monitor_available(spec: dict, opt: str) -> bool:
    """False if the opt's structural_monitor is not a counter in the profile schema."""
    return spec[opt]["structural_monitor"] in PROFILE_COUNTERS


def transform_fired(spec: dict, opt: str, before: dict, after: dict,
                    eps: float = 1e-9) -> bool | None:
    """Did the optimization actually fire? Compares the structural_monitor counter
    before vs after. Returns None if the monitor is unavailable or a counter is missing.

    This is the coverage *evidence* layer: an input whose counter does not move in the
    expected direction did not exercise the optimization (covered nothing real).
    """
    entry = spec[opt]
    mon = entry["structural_monitor"]
    if mon not in PROFILE_COUNTERS:
        return None
    b, a = before.get(mon), after.get(mon)
    if b is None or a is None:
        return None
    exp = entry["structural_expected"]
    if exp == "drop_to_zero":
        return b > eps and a <= eps
    if exp == "decrease":
        return a < b - eps
    if exp == "increase":
        return a > b + eps
    if exp == "ratio":
        return a < b - eps  # a meaningful proportional drop manifests as a decrease
    return None


# ---------------------------------------------------------------- CLI

def _print_all(spec: dict) -> None:
    for opt in opt_ids(spec):
        rep = coverage(spec, opt, spec[opt].get("sample_params", []))
        print(rep.summary())


if __name__ == "__main__":  # pragma: no cover
    import sys
    sp = load_spec(sys.argv[1] if len(sys.argv) > 1 else _SPEC_DEFAULT)
    print(f"loaded {len(opt_ids(sp))} optimizations from spec v{sp['schema_version']}\n")
    _print_all(sp)
