"""
Performance model for Optimization #20: Algebraic Scalar Fusion
================================================================

Pattern: a kernel chains 2+ consecutive ``nisa.tensor_scalar`` (or
``nl.add``/``nl.multiply`` of scalars) calls on the same tile, e.g.::

    t = nisa.tensor_scalar(x, scale=a, op0=multiply)
    t = nisa.tensor_scalar(t, scale=b, op0=add)
    t = nisa.tensor_scalar(t, scale=c, op0=multiply)

These can be algebraically collapsed into a single fused
``nisa.tensor_scalar`` (one ScalarE instruction with combined
coefficients). The pattern shows up frequently in
AdamW-style optimizer updates, learning-rate schedules, and bias
broadcast chains.

opt20 is a "missing model" — it was selected by the autocomp menu on
adamw (per benchmark-opt/reports/mg_v2_predictor_misses.md case C1)
but had no v2 cost model, so model_guided_v2 never surfaced it. v3
ships a minimal ScalarE-instruction-count model to close the gap.

v2 design — caller-anchored
---------------------------
Each consecutive tensor_scalar dispatch contributes one ScalarE
instruction issue. Fusing K consecutive ops into 1 eliminates (K-1)
dispatches per tile. Cost model::

    extra_dispatches = (n_consecutive_ops - 1) * n_tiles
    t_after = alpha_total * t_total_b
            + alpha_dispatch * extra_dispatches   # fits negative — savings
            + alpha_fixed

Initial fit values (alpha_total=1.0, alpha_dispatch=-0.6 µs/dispatch,
alpha_fixed=0) are conservative point estimates from the autocomp
adamw result and one synthetic AdamW recalibration. Recalibration is
Additional patterns require separate modeling — these
constants reproduce the ~1.07x adamw speedup but are NOT a regression
fit yet.

Knobs
-----
- ``has_consecutive_scalar_ops`` (bool): True iff the kernel chains
  2+ tensor_scalar (or scalar add/multiply) ops on the same tile in
  the same iteration. Set false if the chain is already collapsed or
  if intervening non-scalar ops break the chain.
- ``n_consecutive_ops`` (int): length of the longest collapsible chain
  (>= 2). For an AdamW update with 4 chained scalar updates, this is 4.
- ``n_tiles`` (int): outer tile-loop iteration count. Each tile incurs
  the per-chain dispatch overhead.

Required profile fields
-----------------------
``t_total_us``.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, replace

from _common import (  # type: ignore[import-not-found]
    KernelProfile,
    PredictionDelta,
    extrapolation_warning,
)

OPTIMIZATION_NUMBER = 20
OPTIMIZATION_NAME = "Algebraic Scalar Fusion"

REQUIRED_PROFILE_FIELDS = ("t_total_us",)

KNOB_NAMES: dict[str, str] = {
    "has_consecutive_scalar_ops": (
        "True iff the kernel chains 2+ consecutive nisa.tensor_scalar (or "
        "scalar nl.add/nl.multiply) ops on the same tile in the same iter, "
        "with no intervening non-scalar ops that break the chain. Profiler "
        "signature: scalar_engine_active_time is non-trivial AND "
        "vector_engine_instruction_count is dominated by scalar dispatches "
        "(>= 2x the count needed for the kernel's arithmetic intensity)."
    ),
    "n_consecutive_ops": (
        "Length of the longest collapsible scalar-op chain (must be >= 2). "
        "For AdamW's 4-step update (m_t, v_t, m_hat, p update), this is 4. "
        "The optimization replaces n_consecutive_ops dispatches with 1."
    ),
    "n_tiles": (
        "Outer tile-loop iteration count. Each tile pays the per-chain "
        "dispatch overhead once."
    ),
}


@dataclass
class Knobs:
    has_consecutive_scalar_ops: bool
    n_consecutive_ops: int
    n_tiles: int


# opt20 ships with a minimal fit derived from a single calibration point
# (adamw, autocomp result). Stored inline rather than as a fit.json because
# we have only 1 data point — calling load_fit would imply a real regression.
_FIT = {
    "constants": {
        "alpha_total": 1.0,
        "alpha_dispatch": -0.6,  # µs per eliminated ScalarE dispatch
        "alpha_fixed": 0.0,
    },
    "knob_range": {
        "n_consecutive_ops": [2, 6],
        "n_tiles": [1, 256],
    },
    "max_pct_err": None,  # not regressed
    "n_points": 1,
    "fit_source": "single-point conservative estimate; recalibrate with a sweep "
                  "across (n_consecutive_ops, n_tiles) — see "
                  "additional algebraic fusion patterns",
}


def predict(profile: KernelProfile, knobs: Knobs) -> PredictionDelta | None:
    if not knobs.has_consecutive_scalar_ops:
        return None
    if knobs.n_consecutive_ops is None or knobs.n_consecutive_ops < 2:
        return None
    if knobs.n_tiles is None or knobs.n_tiles < 1:
        return None
    if profile.t_total_us is None:
        return None

    alpha_total = _FIT["constants"]["alpha_total"]
    alpha_dispatch = _FIT["constants"]["alpha_dispatch"]
    alpha_fixed = _FIT["constants"]["alpha_fixed"]

    extra_dispatches = (knobs.n_consecutive_ops - 1) * knobs.n_tiles
    t_after = (alpha_total * profile.t_total_us
               + alpha_dispatch * extra_dispatches
               + alpha_fixed)
    t_after = max(t_after, 1.0)

    delta_total = t_after - profile.t_total_us
    predicted_after = replace(profile, t_total_us=t_after)
    return PredictionDelta(
        predicted_after=predicted_after,
        delta_us={"t_total_us": delta_total},
        t_after_us=t_after,
        speedup=profile.t_total_us / t_after if t_after > 0 else float("inf"),
        extrapolation_warning=extrapolation_warning(_FIT, {
            "n_consecutive_ops": knobs.n_consecutive_ops,
            "n_tiles": knobs.n_tiles,
        }),
        notes=(f"Collapses {knobs.n_consecutive_ops} chained scalar ops into 1 "
               f"fused tensor_scalar; eliminates {extra_dispatches} ScalarE "
               f"dispatches over {knobs.n_tiles} tiles. "
               f"NOTE: model is a 1-point conservative estimate; verify "
               f"empirically before relying on the predicted magnitude."),
    )
