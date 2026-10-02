"""
Performance model for Optimization #21: Fuse Two Loops into One
=================================================================

Pattern: a kernel uses two separate ``affine_range`` loops where the
first produces a *small* per-tile intermediate (e.g. row means
[TILE_M, 1] = ~512 bytes) into an HBM buffer; the second reloads
both the input and the intermediate. Total HBM bytes barely change,
but the per-tile DMA *packet* count grows — and the bottleneck is
DMA scheduling overhead, not bandwidth.

Distinct from opt05/07/17 (bandwidth-bound HBM intermediate fusion).
Here the intermediate is too small to register on a byte-anchored
model, but the per-tile packet/scheduling overhead is significant.

v2 design — caller-anchored
---------------------------
Anchor the savings to ``n_loop_boundaries · n_tiles`` (the count of
extra per-tile transitions). This count is the structural quantity
that the optimization eliminates:

    boundary_term = n_loop_boundaries · n_tiles
    t_after = α_total · t_total_b
            + α_boundary · boundary_term      # fits negative — savings
            + α_fixed

Three free parameters; fit by 3x3 LSQ. ``α_boundary`` fits negative
and represents the per-tile-per-boundary scheduling overhead.

Knobs
-----
- ``has_small_hbm_intermediate`` (bool): True iff the kernel uses a
  *small* private_hbm intermediate (profiler shows
  ``spill_save_bytes > 0`` but the spill volume is tiny relative to
  the input — e.g. row means [M,1], not full tiles [M,N]). The
  bottleneck is DMA packet setup overhead, not bandwidth. Set false
  if already fused, or if the intermediate is large (use opt07
  instead).
- ``n_tiles`` (int): tile-loop trip count of each loop (M // TILE_M).
- ``n_loop_boundaries`` (int): inter-loop boundaries to fuse out
  (= n_loops − 1). Typically 1 for a 2-loop kernel.

Required profile fields
-----------------------
``t_total_us``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from _common import (  # type: ignore[import-not-found]
    KernelProfile,
    PredictionDelta,
    extrapolation_warning,
    load_fit,
)

OPTIMIZATION_NUMBER = 21
OPTIMIZATION_NAME = "Fuse Single Loop (Small Intermediate, Packet Overhead)"

REQUIRED_PROFILE_FIELDS = ("t_total_us",)

KNOB_NAMES: dict[str, str] = {
    "has_small_hbm_intermediate": (
        "True iff the kernel uses a *small* private_hbm intermediate "
        "between two affine_range loops (profiler shows "
        "spill_save_bytes > 0 but the spill volume is tiny — e.g. "
        "row means [M,1] = ~bytes, not full tiles [M,N] = ~MB). The "
        "bottleneck is DMA packet setup overhead, not bandwidth. Set "
        "false if already fused, or if the intermediate is large "
        "(use opt07 instead — bandwidth-anchored model)."
    ),
    "n_tiles": (
        "Tile-loop trip count for each loop (M // TILE_M). Each tile "
        "incurs the per-boundary packet overhead."
    ),
    "n_loop_boundaries": (
        "Number of inter-loop boundaries that will be fused out "
        "(= n_loops − 1). Typically 1 for a 2-loop kernel."
    ),
}


@dataclass
class Knobs:
    has_small_hbm_intermediate: bool
    n_tiles: int
    n_loop_boundaries: int = 1


_FIT = load_fit(__file__)


def infer_applicable_from_profile(
    profile: KernelProfile, opt_features: dict
) -> dict | None:
    """v3 profile-driven applicability override.

    opt21 targets *small* private_hbm intermediates whose cost is
    DMA-packet-scheduling overhead, not bandwidth. The profile signature is
    ``spill_save_bytes > 0`` AND a small spill volume relative to total HBM
    traffic. We use spill <= 1% of (hbm_read + hbm_write) as the
    small-intermediate heuristic; n_tiles must be agent-supplied (we cannot
    infer the tile count from profile counters alone).
    """
    if not profile.spill_save_bytes:
        return None
    n_tiles = opt_features.get("n_tiles")
    if not n_tiles or n_tiles < 1:
        return None
    total_hbm = (profile.hbm_read_bytes or 0) + (profile.hbm_write_bytes or 0)
    if total_hbm > 0 and profile.spill_save_bytes > 0.01 * total_hbm:
        return None  # spill is large — opt07 (bandwidth-anchored) is the better fit
    return {
        "has_small_hbm_intermediate": True,
        "n_tiles": n_tiles,
        "n_loop_boundaries": opt_features.get("n_loop_boundaries") or 1,
    }


def predict(profile: KernelProfile, knobs: Knobs) -> PredictionDelta | None:
    if not knobs.has_small_hbm_intermediate:
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
        notes=(f"Fuses {knobs.n_loop_boundaries + 1} loops into 1, "
               f"eliminating {boundary_term} per-tile loop-boundary transitions"),
    )


if __name__ == "__main__":
    import csv
    import pathlib
    csv_path = pathlib.Path(__file__).parent / "calibration.csv"
    if not csv_path.exists():
        print("[error] calibration.csv missing — collect data first")
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
            has_small_hbm_intermediate=True,
            n_tiles=int(b["n_tiles"]),
            n_loop_boundaries=int(b.get("n_loop_boundaries") or 1),
        )
        pred = predict(profile, knobs)
        obs_speedup = float(b["t_total_us"]) / float(a["t_total_us"])
        err = abs(pred.speedup - obs_speedup) / obs_speedup * 100
        print(f"{kid:>30s}  {b['n_tiles']:>7s}  "
              f"{b.get('n_loop_boundaries') or '1':>3s}  "
              f"{pred.speedup:>12.2f}x  {obs_speedup:>11.2f}x  {err:>5.1f}%")
