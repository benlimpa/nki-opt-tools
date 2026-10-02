"""
Performance model for Optimization #17: Keep Data in SBUF Instead of HBM
========================================================================

Pattern: a kernel allocates a per-iteration intermediate with
``buffer=nl.private_hbm``. Each tile iteration writes the intermediate
to HBM and reloads it (or a second loop reloads it), generating
``spill_save_bytes > 0``. Switching the allocation to ``buffer=nl.sbuf``
keeps the intermediate in SBUF for its entire lifetime.

opt17 is **structurally identical** to opt07's loop-fusion model:
both eliminate an HBM round-trip by keeping data in SBUF. v1
calibrated them with consistent T_SPILL_PER_TILE values
(opt07: 1.618 µs, opt17: 1.710 µs, opt05: 1.685 µs). Rather than
re-port with a separate sweep, opt17 reuses opt07's fit constants.

v2 design — caller-anchored
---------------------------
Same formula as opt07 with ``n_intermediates`` typically pinned to 1:

    r_dma   = 1 / (n_intermediates + 1)
    t_after = α_dma · t_dma_b · r_dma
            + α_other · (t_total_b − t_dma_b)
            + α_fixed

Constants come from opt07's ``fit.json``.

Pattern distinction (vs opt05, opt07):
- **opt05**: exactly 2 sequential loops separated by 1 private_hbm.
- **opt07**: ≥3 sequential loops with ≥2 private_hbm intermediates.
- **opt17**: a single loop where the body allocates a private_hbm
  intermediate that's produced and consumed in the *same* loop
  iteration (or carries over). The fix is to allocate in SBUF.

Knobs
-----
- ``has_private_hbm_intermediate`` (bool): True iff the kernel
  allocates at least one ``nl.ndarray`` with ``buffer=nl.private_hbm``
  for a value produced and consumed within the same kernel invocation
  (not a HBM input/output). Profiler signature: ``spill_save_bytes > 0``.
- ``n_intermediates`` (int): number of private_hbm intermediates that
  will be moved to SBUF. Default 1.
- ``n_tiles`` (int): tile-loop trip count. Used for applicability
  + extrapolation only.

Required profile fields
-----------------------
``t_total_us``, ``t_dma_us``.
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

OPTIMIZATION_NUMBER = 17
OPTIMIZATION_NAME = "SBUF Instead of HBM (Eliminate private_hbm Intermediate)"

REQUIRED_PROFILE_FIELDS = ("t_total_us", "t_dma_us")

KNOB_NAMES: dict[str, str] = {
    "has_private_hbm_intermediate": (
        "True iff the kernel allocates at least one nl.ndarray with "
        "buffer=nl.private_hbm for a value produced and consumed "
        "within the same kernel invocation (not an HBM input/output). "
        "Profiler signature: spill_save_bytes > 0 with spill_reload_bytes > 0. "
        "Set false if already using nl.sbuf or nl.psum, or if the "
        "intermediate is too large to fit in SBUF."
    ),
    "n_intermediates": (
        "Number of private_hbm intermediates that will be moved to SBUF. "
        "Typically 1 per private_hbm allocation. Default 1."
    ),
    "n_tiles": (
        "Tile-loop trip count. Used for applicability + extrapolation; "
        "the speedup is governed by the structural ratio rather than "
        "n_tiles. Null if pattern does not apply."
    ),
}


@dataclass
class Knobs:
    has_private_hbm_intermediate: bool
    n_intermediates: int = 1
    n_tiles: int = 1


_FIT_PATH = (pathlib.Path(__file__).parent.parent
             / "opt07-loop-fusion" / "fit.json")
if not _FIT_PATH.exists():
    raise FileNotFoundError(
        f"opt17 reuses opt07's fit; expected {_FIT_PATH} but it is missing."
    )
_FIT = json.loads(_FIT_PATH.read_text())


def infer_applicable_from_profile(
    profile: KernelProfile, opt_features: dict
) -> dict | None:
    """v3 profile-driven applicability override.

    Signal: ``spill_save_bytes > 0`` indicates the compiler materialized a
    private_hbm intermediate that could live in SBUF. Defaults
    ``n_intermediates=1`` and ``n_tiles=1`` if not supplied, matching the
    dataclass defaults (Knobs class fields).
    """
    if not profile.spill_save_bytes:
        return None
    return {
        "has_private_hbm_intermediate": True,
        "n_intermediates": opt_features.get("n_intermediates") or 1,
        "n_tiles": opt_features.get("n_tiles") or 1,
    }


def predict(profile: KernelProfile, knobs: Knobs) -> PredictionDelta | None:
    if not knobs.has_private_hbm_intermediate:
        return None
    if knobs.n_intermediates is None or knobs.n_intermediates < 1:
        return None
    if knobs.n_tiles is None or knobs.n_tiles < 1:
        return None
    if profile.t_dma_us is None:
        return None

    alpha_dma = _FIT["constants"]["alpha_dma"]
    alpha_other = _FIT["constants"]["alpha_other"]
    alpha_fixed = _FIT["constants"]["alpha_fixed"]

    r_dma = 1.0 / (knobs.n_intermediates + 1)
    t_other_b = max(profile.t_total_us - profile.t_dma_us, 0.0)
    t_after = (alpha_dma * profile.t_dma_us * r_dma
               + alpha_other * t_other_b
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
            "n_intermediates": knobs.n_intermediates,
            "n_tiles": knobs.n_tiles,
        }),
        notes=(f"Moves {knobs.n_intermediates} private_hbm intermediate(s) "
               f"to SBUF, eliminating "
               f"{2 * knobs.n_intermediates * knobs.n_tiles} HBM round-trips"),
    )


if __name__ == "__main__":
    # Self-test: re-evaluate against opt07's calibration rows.
    import csv
    csv_path = (pathlib.Path(__file__).parent.parent
                / "opt07-loop-fusion" / "calibration.csv")
    if not csv_path.exists():
        print("[error] opt07 calibration.csv missing — collect data first")
        raise SystemExit(1)
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    by_id: dict[str, dict[str, dict]] = {}
    for r in rows:
        by_id.setdefault(r["kernel_id"], {})[r["variant"]] = r

    print(f"{'kernel_id':>30s}  {'n_int':>5s}  {'n_tiles':>7s}  "
          f"{'pred speedup':>13s}  {'obs speedup':>12s}  {'err%':>6s}")
    print("-" * 90)
    for kid, pair in by_id.items():
        if "before" not in pair or "after" not in pair:
            continue
        b = pair["before"]
        a = pair["after"]
        profile = KernelProfile(
            t_total_us=float(b["t_total_us"]),
            t_active_us=float(b.get("t_active_us") or 0),
            t_dma_us=float(b["t_dma_us"]),
        )
        knobs = Knobs(
            has_private_hbm_intermediate=True,
            n_intermediates=int(b["n_intermediates"]),
            n_tiles=int(b["n_tiles"]),
        )
        pred = predict(profile, knobs)
        obs_speedup = float(b["t_total_us"]) / float(a["t_total_us"])
        err = abs(pred.speedup - obs_speedup) / obs_speedup * 100
        print(f"{kid:>30s}  {b['n_intermediates']:>5s}  {b['n_tiles']:>7s}  "
              f"{pred.speedup:>12.2f}x  {obs_speedup:>11.2f}x  {err:>5.1f}%")
