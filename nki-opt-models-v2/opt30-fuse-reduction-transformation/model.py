"""
Performance model for Optimization #30: Fuse Reduction-Transformation
Loops (Small Intermediate)
=====================================================================

Pattern: a normalization kernel (RMS Norm, layer norm, mean-subtract,
softmax-style) splits the reduction (compute inv_rms, mean, etc.)
and the transformation (apply it) into two separate ``affine_range``
loops, with a *small* ``private_hbm`` intermediate carrying the
reduction result. The compiler already LICMs the redundant input
reload, so the only unique cost is the per-tile DMA *packet*
scheduling overhead — same family as opt21.

opt30 reuses **opt21's fit**: both eliminate the same
packet-overhead structure. v1 cross-validated this — opt30's
calibrated constants matched opt21's at the same scale (≈0.62 µs/
tile/boundary, vs opt21's ≈2.37 — different absolute scales but
same family).

v2 design — caller-anchored
---------------------------
Same formula as opt21:

    boundary_term = n_loop_boundaries · n_tiles
    t_after = α_total · t_total_b
            + α_boundary · boundary_term
            + α_fixed

Pattern distinction (vs opt21):
- **opt21**: any two-loop pattern with a small private_hbm intermediate.
- **opt30**: specifically reduction-then-transformation (variance,
  RMS, mean, softmax) — the intermediate carries a reduction scalar/
  vector that the transformation pass needs to broadcast.

The split exists for the optimizer agent's pattern catalog: opt30's
signature is more specific (matches normalization kernels), opt21
catches the general 2-loop case.

Knobs
-----
- ``has_reduction_transformation_split`` (bool): True iff a
  reduction loop produces a small private_hbm intermediate (the
  reduction scalar/vector) which is reloaded by a separate
  transformation loop. Profiler signature: same as opt21
  (``spill_save_bytes > 0`` with tiny spill volume,
  ``static_dma_packet_count >> 9``). Set false if already fused.
- ``n_tiles`` (int): tile-loop trip count of each loop.
- ``n_loop_boundaries`` (int): inter-loop boundaries to fuse out
  (= n_loops − 1). Default 1.

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

OPTIMIZATION_NUMBER = 30
OPTIMIZATION_NAME = "Fuse Reduction-Transformation Loops (Small Intermediate)"

REQUIRED_PROFILE_FIELDS = ("t_total_us",)

KNOB_NAMES: dict[str, str] = {
    "has_reduction_transformation_split": (
        "True iff the kernel splits a reduction (variance/RMS/mean/"
        "softmax-denom) and the corresponding transformation across "
        "two affine_range loops, with a small private_hbm intermediate "
        "for the reduction result. Profiler signature: "
        "spill_save_bytes > 0 (small, ~KB) AND "
        "static_dma_packet_count >> 9. Set false if already fused, or "
        "if the intermediate is large (use opt07 instead)."
    ),
    "n_tiles": (
        "Tile-loop trip count for each loop (M // TILE_M). Each tile "
        "incurs the per-boundary packet overhead."
    ),
    "n_loop_boundaries": (
        "Number of inter-loop boundaries that will be fused out "
        "(= n_loops − 1). Typically 1 for a reduction+transformation kernel."
    ),
}


@dataclass
class Knobs:
    has_reduction_transformation_split: bool
    n_tiles: int
    n_loop_boundaries: int = 1


_FIT_PATH = (pathlib.Path(__file__).parent.parent
             / "opt21-fuse-single-loop" / "fit.json")
if not _FIT_PATH.exists():
    raise FileNotFoundError(
        f"opt30 reuses opt21's fit; expected {_FIT_PATH} but it is missing."
    )
_FIT = json.loads(_FIT_PATH.read_text())


def infer_applicable_from_profile(
    profile: KernelProfile, opt_features: dict
) -> dict | None:
    """v3 profile-driven applicability override.

    Same family as opt21: small spill + high per-tile packet overhead. Cannot
    distinguish reduction-then-transformation vs generic 2-loop fusion from
    profile alone, so we only auto-apply when the agent explicitly supplied
    ``n_tiles`` (signalling it identified the pattern but forgot to set the
    applicable bit).
    """
    if not profile.spill_save_bytes:
        return None
    n_tiles = opt_features.get("n_tiles")
    if not n_tiles or n_tiles < 1:
        return None
    total_hbm = (profile.hbm_read_bytes or 0) + (profile.hbm_write_bytes or 0)
    if total_hbm > 0 and profile.spill_save_bytes > 0.01 * total_hbm:
        return None  # large spill — use opt07 instead
    return {
        "has_reduction_transformation_split": True,
        "n_tiles": n_tiles,
        "n_loop_boundaries": opt_features.get("n_loop_boundaries") or 1,
    }


def predict(profile: KernelProfile, knobs: Knobs) -> PredictionDelta | None:
    if not knobs.has_reduction_transformation_split:
        return None
    if knobs.n_tiles is None or knobs.n_tiles < 1:
        return None
    if knobs.n_loop_boundaries is None or knobs.n_loop_boundaries < 1:
        return None

    alpha_total = _FIT["constants"]["alpha_total"]
    alpha_boundary = _FIT["constants"]["alpha_boundary"]
    alpha_fixed = _FIT["constants"]["alpha_fixed"]

    boundary_term = knobs.n_loop_boundaries * knobs.n_tiles
    t_after = (alpha_total * profile.t_total_us
               + alpha_boundary * boundary_term
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
            "n_tiles": knobs.n_tiles,
            "n_loop_boundaries": knobs.n_loop_boundaries,
        }),
        notes=(f"Fuses reduction + transformation ({knobs.n_loop_boundaries + 1} "
               f"loops → 1), eliminating {boundary_term} per-tile loop-boundary "
               f"transitions"),
    )


if __name__ == "__main__":
    # Self-test: re-evaluate against opt21's calibration rows (mean-subtract).
    import csv
    csv_path = (pathlib.Path(__file__).parent.parent
                / "opt21-fuse-single-loop" / "calibration.csv")
    if not csv_path.exists():
        print("[error] opt21 calibration.csv missing — collect data first")
        raise SystemExit(1)
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    by_id: dict[str, dict[str, dict]] = {}
    for r in rows:
        by_id.setdefault(r["kernel_id"], {})[r["variant"]] = r

    print(f"{'kernel_id':>30s}  {'n_tiles':>7s}  {'n_b':>3s}  "
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
        )
        knobs = Knobs(
            has_reduction_transformation_split=True,
            n_tiles=int(b["n_tiles"]),
            n_loop_boundaries=int(b.get("n_loop_boundaries") or 1),
        )
        pred = predict(profile, knobs)
        obs_speedup = float(b["t_total_us"]) / float(a["t_total_us"])
        err = abs(pred.speedup - obs_speedup) / obs_speedup * 100
        print(f"{kid:>30s}  {b['n_tiles']:>7s}  "
              f"{b.get('n_loop_boundaries') or '1':>3s}  "
              f"{pred.speedup:>12.2f}x  {obs_speedup:>11.2f}x  {err:>5.1f}%")
