"""
Performance model for Optimization #3: Full Partition Utilization
=================================================================

Pattern: an element-wise (or similarly structured) kernel tiles the partition
axis with ``TILE_P < 128``, using only TILE_P of the 128 NeuronCore
partitions per VectorE op and requiring ``M // TILE_P`` loop iterations.
The fix: set ``TILE_P = 128`` to fully utilize all partitions, reducing the
iteration count by ``128 / TILE_P_current`` and cutting both VectorE
instruction count and DMA packet count proportionally.

v2 design — caller-anchored
---------------------------
The cost decomposes as ``T_per_iter * n_iters + T_overhead``. With
``r = TILE_P_current / 128`` (= n_after / n_before), the overhead-aware
caller-anchored form is:

    t_after = alpha_iter * t_total_b * r + alpha_overhead * (1 - r)

The ``r``-scaled term shrinks the per-iteration share of ``t_total_b``;
the ``(1 - r)`` term lets the irreducible overhead persist (it does not
scale with iteration count). Two free params — fits via 2x2 LSQ.

Knobs
-----
- ``has_underutilized_partition_tiling`` (bool): True iff the kernel loops
  ``range(M // TILE_P)`` with ``TILE_P < 128`` and operates on tiles of
  shape ``[TILE_P, *]`` whose partition axis fits comfortably in SBUF when
  expanded to 128.
- ``M`` (int): total partition-axis dimension (must be divisible by 128).
- ``TILE_P_current`` (int): current sub-128 partition tile size; the
  optimization sets this to 128.

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

OPTIMIZATION_NUMBER = 3
OPTIMIZATION_NAME = "Full Partition Utilization"

REQUIRED_PROFILE_FIELDS = ("t_total_us",)

TILE_P_MAX: int = 128

KNOB_NAMES: dict[str, str] = {
    "has_underutilized_partition_tiling": (
        "True iff the kernel loops sequential_range(M // TILE_P) with "
        "TILE_P < 128 over tiles of shape [TILE_P, *] in SBUF (i.e. the "
        "partition axis is being under-tiled). Set to false (model returns "
        "None) if the kernel already uses TILE_P=128 or doesn't tile the "
        "partition axis this way."
    ),
    "M": (
        "Total partition-axis dimension (the dimension being looped over in "
        "tiles of TILE_P). Read from the kernel's primary ndarray shape "
        "(e.g. M in nl.ndarray((TILE_P, N), ...) inside range(M // TILE_P)). "
        "Must be divisible by 128. Null if not applicable."
    ),
    "TILE_P_current": (
        "Current partition tile size — the divisor in range(M // TILE_P). "
        "Any value < 128 is a symptom of under-utilization. Null if "
        "TILE_P is already 128 or the pattern does not apply."
    ),
}


@dataclass
class Knobs:
    has_underutilized_partition_tiling: bool
    M: int
    TILE_P_current: int


_FIT = load_fit(__file__)


def predict(profile: KernelProfile, knobs: Knobs) -> PredictionDelta | None:
    if not knobs.has_underutilized_partition_tiling:
        return None
    if knobs.M is None or knobs.TILE_P_current is None:
        return None
    if knobs.TILE_P_current >= TILE_P_MAX:
        return None
    if knobs.M % TILE_P_MAX != 0 or knobs.M % knobs.TILE_P_current != 0:
        return None
    if profile.t_total_us is None:
        return None

    alpha_iter = _FIT["constants"]["alpha_iter"]
    alpha_overhead = _FIT["constants"]["alpha_overhead"]

    r = knobs.TILE_P_current / TILE_P_MAX
    t_after = alpha_iter * profile.t_total_us * r + alpha_overhead * (1.0 - r)
    t_after = max(t_after, 1.0)

    delta_total = t_after - profile.t_total_us
    predicted_after = replace(
        profile,
        t_total_us=t_after,
    )
    n_before = knobs.M // knobs.TILE_P_current
    n_after = knobs.M // TILE_P_MAX
    return PredictionDelta(
        predicted_after=predicted_after,
        delta_us={"t_total_us": delta_total},
        t_after_us=t_after,
        speedup=profile.t_total_us / t_after if t_after > 0 else float("inf"),
        extrapolation_warning=extrapolation_warning(_FIT, {
            "M": knobs.M,
            "TILE_P_current": knobs.TILE_P_current,
        }),
        notes=(f"Reduces partition-axis iterations from {n_before} "
               f"(TILE_P={knobs.TILE_P_current}) to {n_after} (TILE_P=128); "
               f"VectorE ops and DMA packets shrink by {n_before // n_after}x"),
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

    print(f"{'kernel_id':>30s}  {'M':>5s}  {'TILE_P':>6s}  "
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
            has_underutilized_partition_tiling=True,
            M=int(b["M"]),
            TILE_P_current=int(b["TILE_P_current"]),
        )
        pred = predict(profile, knobs)
        obs_speedup = float(b["t_total_us"]) / float(a["t_total_us"])
        err = abs(pred.speedup - obs_speedup) / obs_speedup * 100
        print(f"{kid:>30s}  {b['M']:>5s}  {b['TILE_P_current']:>6s}  "
              f"{pred.speedup:>12.2f}x  {obs_speedup:>11.2f}x  {err:>5.1f}%")
