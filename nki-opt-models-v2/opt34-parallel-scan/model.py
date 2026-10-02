"""
Performance model for Optimization #34: Parallel Scan with MatMul Carry-over
=============================================================================

Pattern: a 1D associative scan (cumsum, cumprod, ...) over ``N_total`` elements
is computed sequentially, one partition-row at a time, via
``sequential_range(P)``. The optimized version reshapes to ``[P, TILE_N]``,
scans all P partitions in parallel, then uses a single ``nc_matmul`` with a
strictly upper-triangular matrix to compute cross-row carry offsets in one
TensorE call. P is the NeuronCore partition count (hardware constant = 128).

v2 design — caller-anchored
---------------------------
The optimized kernel keeps DMA active time roughly proportional to bytes moved
(slightly faster due to packet coalescing) and parallelizes the rest by P-fold:

    t_after = beta_dma * t_dma_b
            + beta_par * (t_total_b - t_dma_b) / P
            + alpha_fixed

The user's measured ``t_dma_us`` and ``t_total_us`` anchor the scaling laws;
``N_total`` enters only via applicability/extrapolation (and the fixed P=128
hardware constraint).

Knobs
-----
- ``has_sequential_scan`` (bool): True iff the kernel uses a
  ``sequential_range`` 1D scan with row-by-row carry. Set to false (model
  returns None) if not.
- ``N_total`` (int): total elements scanned. Must be divisible by P=128.

Required profile fields
-----------------------
``t_total_us``, ``t_dma_us``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from _common import (  # type: ignore[import-not-found]
    KernelProfile,
    PredictionDelta,
    extrapolation_warning,
    load_fit,
)

OPTIMIZATION_NUMBER = 34
OPTIMIZATION_NAME = "Parallel Scan with MatMul Carry-over"

REQUIRED_PROFILE_FIELDS = ("t_total_us", "t_dma_us")

P_HARDWARE = 128

KNOB_NAMES: dict[str, str] = {
    "has_sequential_scan": (
        "True iff the kernel runs a row-by-row 1D scan via "
        "sequential_range(P) with a per-iteration carry-over (e.g., "
        "nisa.tensor_tensor_scan with `initial=carry_sb`). Set to false "
        "if the scan is already parallelized or the kernel is not a 1D scan."
    ),
    "N_total": (
        "Total number of elements being scanned (e.g., the length of the 1D "
        "input array). Must be divisible by 128 (hardware partition count) "
        "for the optimization to apply. Null if not applicable."
    ),
}


@dataclass
class Knobs:
    has_sequential_scan: bool
    N_total: int


_FIT = load_fit(__file__)


def predict(profile: KernelProfile, knobs: Knobs) -> PredictionDelta | None:
    if not knobs.has_sequential_scan:
        return None
    if knobs.N_total is None or knobs.N_total <= 0:
        return None
    if knobs.N_total % P_HARDWARE != 0:
        return None
    if profile.t_dma_us is None:
        return None

    beta_dma = _FIT["constants"]["beta_dma"]
    beta_par = _FIT["constants"]["beta_par"]
    alpha_fixed = _FIT["constants"]["alpha_fixed"]

    t_serial = max(profile.t_total_us - profile.t_dma_us, 0.0)
    t_after = (beta_dma * profile.t_dma_us
               + beta_par * t_serial / P_HARDWARE
               + alpha_fixed)
    # Floor at a small positive value (DMA bulk transfer of N_total floats)
    t_after = max(t_after, profile.t_dma_us * 0.25)

    delta_total = t_after - profile.t_total_us
    predicted_after = replace(
        profile,
        t_total_us=t_after,
        # Vector engine time gets divided by ~P; approximate.
        t_vector_us=(profile.t_vector_us / P_HARDWARE) if profile.t_vector_us else None,
        t_active_us=t_after,
    )
    return PredictionDelta(
        predicted_after=predicted_after,
        delta_us={"t_total_us": delta_total},
        t_after_us=t_after,
        speedup=profile.t_total_us / t_after if t_after > 0 else float("inf"),
        extrapolation_warning=extrapolation_warning(_FIT, {"N_total": knobs.N_total}),
        notes=(f"Parallelizes serial scan across {P_HARDWARE} partitions; "
               f"TILE_N = {knobs.N_total // P_HARDWARE}"),
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

    print(f"{'kernel_id':>28s}  {'N_total':>8s}  {'pred speedup':>13s}  "
          f"{'obs speedup':>12s}  {'err%':>6s}")
    print("-" * 80)
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
        knobs = Knobs(has_sequential_scan=True, N_total=int(b["N_total"]))
        pred = predict(profile, knobs)
        obs_speedup = float(b["t_total_us"]) / float(a["t_total_us"])
        err = abs(pred.speedup - obs_speedup) / obs_speedup * 100
        print(f"{kid:>28s}  {b['N_total']:>8s}  {pred.speedup:>12.2f}x  "
              f"{obs_speedup:>11.2f}x  {err:>5.1f}%")
