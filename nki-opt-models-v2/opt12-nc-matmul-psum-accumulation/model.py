"""
Performance model for Optimization #12: nc_matmul PSUM Accumulation Group
=========================================================================

Pattern: a matmul kernel reduces over K-tiles using ``sequential_range``,
forcing each K-tile's PSUM result to be evicted to SBUF and VectorE-added
to a running accumulator. Switching to ``affine_range`` lets the compiler
form a PSUM accumulation group: TensorE accumulates partials in PSUM and
emits a single SBUF eviction per M-tile, eliminating
``(n_k_tiles - 1)`` per-M-tile evictions + ``n_k_tiles`` VectorE adds.

v2 design — caller-anchored
---------------------------
Each eliminated eviction saves a fixed amount of time. Total savings
scale with ``extra_evictions = n_m_tiles * (n_k_tiles - 1)``. The rest
of the kernel cost is preserved.

    t_after = alpha_total * t_total_b
              + alpha_evict * extra_evictions
              + alpha_fixed

Three free parameters; fit by 3x3 LSQ. ``alpha_evict`` is expected to
be negative.

Knobs
-----
- ``uses_sequential_range_for_k`` (bool): True iff the K-tile reduction
  loop uses ``sequential_range`` (forcing per-iteration eviction). Set
  to false (model returns None) if already using ``affine_range``.
- ``n_m_tiles`` (int): number of M-tiles (= ``M // TILE_M``).
- ``n_k_tiles`` (int): number of K-tiles per M-tile (= ``K // TILE_K``);
  must be ≥ 2.

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

OPTIMIZATION_NUMBER = 12
OPTIMIZATION_NAME = "nc_matmul PSUM Accumulation Group"

REQUIRED_PROFILE_FIELDS = ("t_total_us",)

KNOB_NAMES: dict[str, str] = {
    "uses_sequential_range_for_k": (
        "True iff the K-tile reduction loop uses nl.sequential_range, "
        "which forces a PSUM→SBUF eviction + VectorE add per K-tile. "
        "Set to false if the K-loop already uses nl.affine_range "
        "(compiler forms a PSUM accumulation group automatically)."
    ),
    "n_m_tiles": (
        "Number of M-tiles in the outer loop (= M // TILE_M). Null if "
        "the pattern does not apply."
    ),
    "n_k_tiles": (
        "Number of K-tiles per M-tile in the inner reduction loop "
        "(= K // TILE_K). Must be ≥ 2 for the optimization to have "
        "any effect. Null if the pattern does not apply."
    ),
}


@dataclass
class Knobs:
    uses_sequential_range_for_k: bool
    n_m_tiles: int
    n_k_tiles: int


_FIT = load_fit(__file__)


def predict(profile: KernelProfile, knobs: Knobs) -> PredictionDelta | None:
    if not knobs.uses_sequential_range_for_k:
        return None
    if knobs.n_m_tiles is None or knobs.n_k_tiles is None:
        return None
    if knobs.n_m_tiles <= 0 or knobs.n_k_tiles < 2:
        return None
    if profile.t_total_us is None:
        return None

    alpha_total = _FIT["constants"]["alpha_total"]
    alpha_evict = _FIT["constants"]["alpha_evict"]
    alpha_fixed = _FIT["constants"]["alpha_fixed"]

    extra_evictions = knobs.n_m_tiles * (knobs.n_k_tiles - 1)
    t_after = (alpha_total * profile.t_total_us
               + alpha_evict * extra_evictions
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
            "n_m_tiles": knobs.n_m_tiles,
            "n_k_tiles": knobs.n_k_tiles,
        }),
        notes=(f"Eliminates {extra_evictions} PSUM→SBUF evictions and "
               f"{knobs.n_m_tiles * knobs.n_k_tiles} VectorE adds by "
               f"switching K-loop to affine_range"),
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

    print(f"{'kernel_id':>30s}  {'n_m':>4s}  {'n_k':>4s}  "
          f"{'pred speedup':>13s}  {'obs speedup':>12s}  {'err%':>6s}")
    print("-" * 80)
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
            uses_sequential_range_for_k=True,
            n_m_tiles=int(b["n_m_tiles"]),
            n_k_tiles=int(b["n_k_tiles"]),
        )
        pred = predict(profile, knobs)
        obs_speedup = float(b["t_total_us"]) / float(a["t_total_us"])
        err = abs(pred.speedup - obs_speedup) / obs_speedup * 100
        print(f"{kid:>30s}  {b['n_m_tiles']:>4s}  {b['n_k_tiles']:>4s}  "
              f"{pred.speedup:>12.2f}x  {obs_speedup:>11.2f}x  {err:>5.1f}%")
