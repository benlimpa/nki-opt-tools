"""
Performance model for Optimization #6: Skip Masked (Padding) Computation
========================================================================

Pattern: a tile loop iterates over ``total_tiles`` even though only
``valid_tiles < total_tiles`` contain real data — the padding tiles do
useless work. The fix bounds the loop to ``valid_tiles`` (a Python int
known at trace time), eliminating the wasted iterations.

v2 design — caller-anchored
---------------------------
With ``r = valid_tiles / total_tiles``, the overhead-aware caller-anchored
form is the same shape as opt03/opt11:

    t_after = alpha_iter * t_total_b * r + alpha_overhead * (1 - r)

The ``r``-scaled term shrinks the per-iteration share of ``t_total_b``;
the ``(1 - r)`` term lets the irreducible overhead persist across the
shorter loop. Two free params — fits via 2x2 LSQ.

Knobs
-----
- ``has_padding_iterations`` (bool): True iff the kernel iterates over
  tiles where some are padding (compile-time-known valid count).
- ``total_tiles`` (int): current loop bound.
- ``valid_tiles`` (int): non-padding tile count (must be < total_tiles).

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

OPTIMIZATION_NUMBER = 6
OPTIMIZATION_NAME = "Skip Masked Compute"

REQUIRED_PROFILE_FIELDS = ("t_total_us",)

KNOB_NAMES: dict[str, str] = {
    "has_padding_iterations": (
        "True iff the kernel's tile loop iterates over more tiles than "
        "contain real data — i.e. the upper bound includes padding tiles "
        "whose output is unused. Set to false (model returns None) if "
        "every tile holds valid data."
    ),
    "total_tiles": (
        "Current loop bound — the range argument of the tile loop "
        "(typically M // TILE_P). Null if the pattern does not apply."
    ),
    "valid_tiles": (
        "Number of tiles containing real (non-padding) data. Must be a "
        "Python int known at trace time and strictly less than "
        "total_tiles. Null if all tiles are valid."
    ),
}


@dataclass
class Knobs:
    has_padding_iterations: bool
    total_tiles: int
    valid_tiles: int


_FIT = load_fit(__file__)


def predict(profile: KernelProfile, knobs: Knobs) -> PredictionDelta | None:
    if not knobs.has_padding_iterations:
        return None
    if knobs.total_tiles is None or knobs.valid_tiles is None:
        return None
    if knobs.valid_tiles <= 0 or knobs.valid_tiles >= knobs.total_tiles:
        return None
    if profile.t_total_us is None:
        return None

    alpha_iter = _FIT["constants"]["alpha_iter"]
    alpha_overhead = _FIT["constants"]["alpha_overhead"]

    r = knobs.valid_tiles / knobs.total_tiles
    t_after = alpha_iter * profile.t_total_us * r + alpha_overhead * (1.0 - r)
    t_after = max(t_after, 1.0)

    delta_total = t_after - profile.t_total_us
    predicted_after = replace(profile, t_total_us=t_after)
    return PredictionDelta(
        predicted_after=predicted_after,
        delta_us={"t_total_us": delta_total},
        t_after_us=t_after,
        speedup=profile.t_total_us / t_after if t_after > 0 else float("inf"),
        extrapolation_warning=extrapolation_warning(_FIT, {
            "total_tiles": knobs.total_tiles,
            "valid_tiles": knobs.valid_tiles,
        }),
        notes=(f"Reduces tile loop from {knobs.total_tiles} to "
               f"{knobs.valid_tiles} iterations "
               f"({(1 - r) * 100:.0f}% of work skipped)"),
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

    print(f"{'kernel_id':>30s}  {'tot':>4s}  {'val':>4s}  "
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
            has_padding_iterations=True,
            total_tiles=int(b["total_tiles"]),
            valid_tiles=int(b["valid_tiles"]),
        )
        pred = predict(profile, knobs)
        obs_speedup = float(b["t_total_us"]) / float(a["t_total_us"])
        err = abs(pred.speedup - obs_speedup) / obs_speedup * 100
        print(f"{kid:>30s}  {b['total_tiles']:>4s}  {b['valid_tiles']:>4s}  "
              f"{pred.speedup:>12.2f}x  {obs_speedup:>11.2f}x  {err:>5.1f}%")
