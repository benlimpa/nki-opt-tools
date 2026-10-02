"""
Performance model for Optimization #18: Stronger K-Tiling
==========================================================

Pattern: a matmul kernel tiles the K (contraction) dimension into small
``TILE_K`` chunks (e.g. ``TILE_K=32``) and runs ``K // TILE_K`` nc_matmul
calls per M-tile. Increasing ``TILE_K`` toward 128 (the nc_matmul partition
maximum) reduces the number of nc_matmul calls proportionally and — critically
— lets the compiler hoist the moving (rhs) tensor loads outside the M-tile
loop via LICM, slashing HBM reads.

v2 design — caller-anchored
---------------------------
TensorE active time scales as ``1 / tile_k_ratio``; DMA active time also
scales (compounding instruction reduction with the LICM speedup):

    t_after = alpha_tensor * t_tensor_b / tile_k_ratio
            + alpha_dma    * t_dma_b    / tile_k_ratio
            + alpha_other  * (t_total_b - t_tensor_b - t_dma_b)
            + alpha_fixed

``tile_k_ratio = TILE_K_target / TILE_K_current`` (typically 128/TILE_K_current).

Knobs
-----
- ``has_fine_k_tiling`` (bool): True iff the kernel inner-loops over
  ``K // TILE_K`` chunks with TILE_K < 128 in an nc_matmul accumulation.
- ``K`` (int): full contraction-dim size.
- ``TILE_K_current`` (int): kernel's current K-tile size. Must divide K.

Required profile fields
-----------------------
``t_total_us``, ``t_tensor_us``, ``t_dma_us``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from _common import (  # type: ignore[import-not-found]
    KernelProfile,
    PredictionDelta,
    extrapolation_warning,
    load_fit,
)

OPTIMIZATION_NUMBER = 18
OPTIMIZATION_NAME = "Stronger K-Tiling"

REQUIRED_PROFILE_FIELDS = ("t_total_us", "t_tensor_us", "t_dma_us")

TILE_K_TARGET = 128  # nc_matmul partition maximum

KNOB_NAMES: dict[str, str] = {
    "has_fine_k_tiling": (
        "True iff the kernel runs an inner loop over K-tiles (e.g. "
        "`for k in affine_range(K // TILE_K)`) with TILE_K < 128 inside an "
        "nc_matmul accumulation pattern. Set to false otherwise."
    ),
    "K": (
        "Full contraction-axis size of the matmul (e.g. inner dim of A.T @ B). "
        "Must be ≥ 128 and divisible by 128 for the optimization to apply. "
        "Null if not applicable."
    ),
    "TILE_K_current": (
        "Kernel's current K-tile size. The optimization replaces it with "
        "TILE_K=128 (or a smaller divisor of K if K < 128). Null if "
        "not applicable."
    ),
}


@dataclass
class Knobs:
    has_fine_k_tiling: bool
    K: int
    TILE_K_current: int


_FIT = load_fit(__file__)


def predict(profile: KernelProfile, knobs: Knobs) -> PredictionDelta | None:
    if not knobs.has_fine_k_tiling:
        return None
    if knobs.K is None or knobs.TILE_K_current is None:
        return None
    if knobs.TILE_K_current >= TILE_K_TARGET:
        return None  # already optimal
    if knobs.K % TILE_K_TARGET != 0 or knobs.K % knobs.TILE_K_current != 0:
        return None
    if profile.t_tensor_us is None or profile.t_dma_us is None:
        return None

    tile_k_ratio = TILE_K_TARGET / knobs.TILE_K_current

    a_t = _FIT["constants"]["alpha_tensor"]
    a_d = _FIT["constants"]["alpha_dma"]
    a_o = _FIT["constants"]["alpha_other"]
    a_f = _FIT["constants"]["alpha_fixed"]

    t_other = max(profile.t_total_us - profile.t_tensor_us - profile.t_dma_us, 0.0)
    t_after = (a_t * profile.t_tensor_us / tile_k_ratio
               + a_d * profile.t_dma_us / tile_k_ratio
               + a_o * t_other
               + a_f)
    t_after = max(t_after, profile.t_dma_us / tile_k_ratio * 0.5)

    delta_total = t_after - profile.t_total_us
    predicted_after = replace(
        profile,
        t_total_us=t_after,
        t_tensor_us=profile.t_tensor_us / tile_k_ratio,
        t_dma_us=profile.t_dma_us / tile_k_ratio,
        t_active_us=t_after,
    )
    return PredictionDelta(
        predicted_after=predicted_after,
        delta_us={
            "t_total_us": delta_total,
            "t_tensor_us": profile.t_tensor_us / tile_k_ratio - profile.t_tensor_us,
            "t_dma_us": profile.t_dma_us / tile_k_ratio - profile.t_dma_us,
        },
        t_after_us=t_after,
        speedup=profile.t_total_us / t_after if t_after > 0 else float("inf"),
        extrapolation_warning=extrapolation_warning(_FIT, {
            "K": knobs.K,
            "tile_k_ratio": tile_k_ratio,
        }),
        notes=(f"TILE_K {knobs.TILE_K_current} → {TILE_K_TARGET}: "
               f"{tile_k_ratio:.0f}× fewer nc_matmul calls + compiler LICM hoist of rhs"),
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

    print(f"{'kernel_id':>30s}  {'K':>4s}  {'TILE_K':>6s}  {'pred speedup':>13s}  "
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
            t_dma_us=float(b["t_dma_us"]),
        )
        knobs = Knobs(
            has_fine_k_tiling=True,
            K=int(b["K"]),
            TILE_K_current=int(b["TILE_K_current"]),
        )
        pred = predict(profile, knobs)
        obs_speedup = float(b["t_total_us"]) / float(a["t_total_us"])
        err = abs(pred.speedup - obs_speedup) / obs_speedup * 100
        print(f"{kid:>30s}  {b['K']:>4s}  {b['TILE_K_current']:>6s}  "
              f"{pred.speedup:>12.2f}x  {obs_speedup:>11.2f}x  {err:>5.1f}%")
