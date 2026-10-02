"""
Performance model for Optimization #37: DMA Transpose
======================================================

Pattern: a kernel uses ``nisa.nc_transpose`` (Tensor Engine matmul-with-
identity) to transpose a [D0, D3] tile in SBUF, repeated ``num_iters`` times.
Replacing each call with ``nisa.dma_transpose`` (which transposes during the
HBM→SBUF transfer) eliminates the TensorE work entirely.

v2 design — caller-anchored
---------------------------
The optimized kernel preserves most non-TensorE overhead and removes the
TensorE active time:

    t_after = alpha * (t_total_b - t_tensor_b) + alpha_fixed

The user's measured ``t_total_us`` and ``t_tensor_us`` carry the absolute
scale; ``num_iters`` and shape only enter via applicability/extrapolation.

Knobs
-----
- ``has_nc_transpose`` (bool): True iff the kernel calls ``nisa.nc_transpose``
  to transpose an SBUF tile that was just loaded from HBM (i.e. the
  load+transpose can be fused into a single ``dma_transpose``).
- ``num_iters`` (int): how many times the transpose runs (e.g. inside an
  outer loop). 1 if there is no outer loop.
- ``D0`` (int): partition-axis size of the transposed tile (typically 128).
- ``D3`` (int): free-axis size of the transposed tile.

Required profile fields
-----------------------
``t_total_us``, ``t_tensor_us``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from _common import (  # type: ignore[import-not-found]
    KernelProfile,
    PredictionDelta,
    extrapolation_warning,
    load_fit,
)

OPTIMIZATION_NUMBER = 37
OPTIMIZATION_NAME = "DMA Transpose"

REQUIRED_PROFILE_FIELDS = ("t_total_us", "t_tensor_us")

KNOB_NAMES: dict[str, str] = {
    "has_nc_transpose": (
        "True iff the kernel calls nisa.nc_transpose on an SBUF tile that was "
        "just loaded from HBM (the load+transpose can be fused into a single "
        "nisa.dma_transpose). Set to false if no such pattern."
    ),
    "num_iters": (
        "How many times the transpose runs (e.g. range argument of the "
        "enclosing affine_range/sequential_range). 1 if no outer loop. "
        "Null if the pattern does not apply."
    ),
    "D0": (
        "Partition-axis size of the transposed tile (≤128, typically 128). "
        "Null if not applicable."
    ),
    "D3": (
        "Free-axis size of the transposed tile (the columns being swapped "
        "with rows). Null if not applicable."
    ),
}


@dataclass
class Knobs:
    has_nc_transpose: bool
    num_iters: int
    D0: int
    D3: int


_FIT = load_fit(__file__)


def predict(profile: KernelProfile, knobs: Knobs) -> PredictionDelta | None:
    if not knobs.has_nc_transpose:
        return None
    if knobs.num_iters is None or knobs.num_iters < 1:
        return None
    if profile.t_tensor_us is None:
        return None

    alpha = _FIT["constants"]["alpha"]
    alpha_fixed = _FIT["constants"]["alpha_fixed"]

    t_non_tensor = max(profile.t_total_us - profile.t_tensor_us, 0.0)
    t_after = alpha * t_non_tensor + alpha_fixed
    t_after = max(t_after, 1.0)

    delta_total = t_after - profile.t_total_us
    predicted_after = replace(
        profile,
        t_total_us=t_after,
        t_tensor_us=0.0,
        t_active_us=t_non_tensor,
    )
    return PredictionDelta(
        predicted_after=predicted_after,
        delta_us={
            "t_total_us": delta_total,
            "t_tensor_us": -profile.t_tensor_us,
        },
        t_after_us=t_after,
        speedup=profile.t_total_us / t_after if t_after > 0 else float("inf"),
        extrapolation_warning=extrapolation_warning(_FIT, {
            "num_iters": knobs.num_iters,
            "D0": knobs.D0,
            "D3": knobs.D3,
        }),
        notes=(f"Replaces {knobs.num_iters} nc_transpose calls with "
               f"dma_transpose (eliminates TensorE active time)"),
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

    print(f"{'kernel_id':>30s}  {'iters':>5s}  {'D0xD3':>9s}  {'pred speedup':>13s}  "
          f"{'obs speedup':>12s}  {'err%':>6s}")
    print("-" * 90)
    for kid, pair in by_id.items():
        if "before" not in pair or "after" not in pair:
            continue
        b = pair["before"]
        a = pair["after"]
        profile = KernelProfile(
            t_total_us=float(b["t_total_us"]),
            t_active_us=float(b.get("t_active_us") or 0),
            t_tensor_us=float(b["t_tensor_us"]),
        )
        knobs = Knobs(
            has_nc_transpose=True,
            num_iters=int(b["num_iters"]),
            D0=int(b["D0"]),
            D3=int(b["D3"]),
        )
        pred = predict(profile, knobs)
        obs_speedup = float(b["t_total_us"]) / float(a["t_total_us"])
        err = abs(pred.speedup - obs_speedup) / obs_speedup * 100
        print(f"{kid:>30s}  {b['num_iters']:>5s}  {b['D0']}x{b['D3']:<5}  "
              f"{pred.speedup:>12.2f}x  {obs_speedup:>11.2f}x  {err:>5.1f}%")
