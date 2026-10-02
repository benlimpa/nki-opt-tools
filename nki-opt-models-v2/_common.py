"""
nki-opt-models-v2/_common.py
============================

Shared types and helpers for v2 performance models.

Two key types:

- ``KernelProfile`` — measured profile of a kernel run. Always supplied by the
  caller (the optimizer agent). Each field is the corresponding neuron-profile
  metric in **microseconds** (times) or raw counts (bytes/instructions).

- ``PredictionDelta`` — a model's prediction of how a kernel's profile will
  change after applying its optimization. Returned by every model's
  ``predict()`` function.

The split is deliberate: v1 conflated cost-scaling laws with hardcoded baseline
anchors. v2 keeps the scaling laws in code and uses the caller's measured
profile as the anchor.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field, fields, replace
from typing import Any


# ---------------------------------------------------------------------------
# KernelProfile
# ---------------------------------------------------------------------------

@dataclass
class KernelProfile:
    """A measured kernel profile, normalized to microseconds and raw counts."""

    # Required (always present from neuron-profile)
    t_total_us: float
    t_active_us: float

    # Engine active times (us). None if profiler did not report.
    t_tensor_us: float | None = None
    t_vector_us: float | None = None
    t_scalar_us: float | None = None
    t_gpsimd_us: float | None = None
    t_dma_us: float | None = None
    t_static_dma_us: float | None = None

    # Counters
    hbm_read_bytes: int | None = None
    hbm_write_bytes: int | None = None
    dma_packet_count: int | None = None
    static_dma_packet_count: int | None = None
    vector_instr_count: int | None = None
    tensor_instr_count: int | None = None
    matmul_instr_count: int | None = None

    # Spill counters (v3): non-zero indicates the compiler materialized an
    # intermediate to HBM (a private_hbm round-trip). Used as a profile-driven
    # gate for the fusion opts (opt05/07/17/21/30) in v3 so the agent does not
    # have to extract has_hbm_intermediate from source.
    spill_save_bytes: int | None = None
    spill_reload_bytes: int | None = None

    @classmethod
    def from_metrics(cls, metrics: dict[str, Any]) -> "KernelProfile":
        """
        Build a KernelProfile from the dict returned by
        ``opt_profiler.NKIOptimizationProfiler._benchmark_neff()``.

        That dict has:
        - ``latency_ms`` (ms, total wall time)
        - ``total_active_time_ms`` (ms, sum of engine busy)
        - All other neuron-profile fields verbatim — engine active times in
          **seconds**, byte counts as ints.
        """
        def _us_from_ms(key: str) -> float | None:
            v = metrics.get(key)
            return None if v is None else float(v) * 1e3

        def _us_from_sec(key: str) -> float | None:
            v = metrics.get(key)
            return None if v is None else float(v) * 1e6

        def _int(key: str) -> int | None:
            v = metrics.get(key)
            return None if v is None else int(v)

        # Total/active are returned in ms by the profiler adapter
        t_total_us = _us_from_ms("latency_ms")
        t_active_us = _us_from_ms("total_active_time_ms")
        if t_total_us is None:
            raise ValueError("metrics dict missing required field 'latency_ms'")
        if t_active_us is None:
            # Fall back to summing engine actives, but most callers will have it
            t_active_us = 0.0

        # Try dynamic_dma_packet_count first; fall back to deriving from percent
        dma_packet_count = _int("dynamic_dma_packet_count")
        if dma_packet_count is None:
            dma_packet_count = _int("dma_packet_count")

        return cls(
            t_total_us=t_total_us,
            t_active_us=t_active_us,
            t_tensor_us=_us_from_sec("tensor_engine_active_time"),
            t_vector_us=_us_from_sec("vector_engine_active_time"),
            t_scalar_us=_us_from_sec("scalar_engine_active_time"),
            t_gpsimd_us=_us_from_sec("gpsimd_engine_active_time"),
            t_dma_us=_us_from_sec("dma_active_time"),
            t_static_dma_us=_us_from_sec("static_dma_active_time"),
            hbm_read_bytes=_int("hbm_read_bytes"),
            hbm_write_bytes=_int("hbm_write_bytes"),
            dma_packet_count=dma_packet_count,
            static_dma_packet_count=_int("static_dma_packet_count"),
            vector_instr_count=_int("vector_engine_instruction_count"),
            tensor_instr_count=_int("tensor_engine_instruction_count"),
            matmul_instr_count=_int("matmul_instruction_count"),
            spill_save_bytes=_int("spill_save_bytes"),
            spill_reload_bytes=_int("spill_reload_bytes"),
        )

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "KernelProfile":
        """Build a KernelProfile from a flat dict of v2-named fields (us, counts)."""
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid})

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


# ---------------------------------------------------------------------------
# PredictionDelta
# ---------------------------------------------------------------------------

@dataclass
class PredictionDelta:
    """A model's predicted post-optimization profile and headline numbers."""

    predicted_after: KernelProfile
    delta_us: dict[str, float] = field(default_factory=dict)
    t_after_us: float = 0.0
    speedup: float = 1.0
    extrapolation_warning: str | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicted_after": self.predicted_after.to_dict(),
            "delta_us": self.delta_us,
            "t_after_us": self.t_after_us,
            "speedup": self.speedup,
            "extrapolation_warning": self.extrapolation_warning,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Fit loading + extrapolation helpers
# ---------------------------------------------------------------------------

def load_fit(model_file: str) -> dict[str, Any]:
    """
    Load ``fit.json`` from the directory containing ``model_file`` (typically
    invoked as ``load_fit(__file__)``).

    The JSON shape is:
        {
          "constants": {"alpha_dma": 1.01, "alpha_overhead": 0.85, ...},
          "knob_range": {"n_tiles": [4, 32], ...},
          "max_pct_err": 4.1,
          "n_points": 6
        }

    Returns the parsed dict. Raises FileNotFoundError with a clear message if
    fit.json is missing — that means the model needs ``python fit.py`` run.
    """
    path = pathlib.Path(model_file).parent / "fit.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python {path.parent}/fit.py` to generate it "
            f"from calibration.csv."
        )
    return json.loads(path.read_text())


def extrapolation_warning(
    fit: dict[str, Any],
    knobs: dict[str, float | int],
) -> str | None:
    """
    Return a human-readable warning if any knob is outside the calibrated range.

    The returned string lists every out-of-range knob; returns None when all
    knobs are inside the calibrated range.
    """
    knob_range = fit.get("knob_range", {})
    out_of_range = []
    for name, value in knobs.items():
        rng = knob_range.get(name)
        if rng is None or value is None:
            continue
        lo, hi = rng
        if value < lo or value > hi:
            out_of_range.append(f"{name}={value} outside calibrated range [{lo}, {hi}]")
    if not out_of_range:
        return None
    return "; ".join(out_of_range)
