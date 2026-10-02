"""
Performance model for Optimization #16: Pipeline Compute and DMA (3-Stage)
==========================================================================

Pattern: a tiled matmul kernel serializes load → compute → store per tile. A
3-stage software pipeline overlaps the three phases across consecutive tiles
using ping-pong buffers. When DMA is the bottleneck (t_dma > t_tensor), the
pipeline hides compute fully behind DMA.

v2 design — caller-anchored
---------------------------
Unlike v1, this model does NOT predict an absolute baseline; it takes the
caller's measured profile as the anchor and predicts how the bottleneck shifts.

    t_after = alpha_dma * t_dma + alpha_overhead * (t_total - t_dma - t_tensor)

The two alphas are fit from calibration.csv against multiple (n_tiles, dtype,
TILE_M) variants. The model only ships when ≥2 paired observations are present;
``alpha_dma`` should be ≈ 1 and ``alpha_overhead`` ≈ 0.85 (the pipeline reduces
scheduling-overhead but does not add to DMA active time).

Knobs
-----
- ``n_tiles`` (int): pipeline depth; pattern requires ≥3.
- ``has_serial_dma_compute`` (bool): True iff the kernel currently serializes
  load/compute/store per tile (not already pipelined).

Required profile fields
-----------------------
``t_total_us``, ``t_dma_us``, ``t_tensor_us``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from _common import (  # type: ignore[import-not-found]
    KernelProfile,
    PredictionDelta,
    extrapolation_warning,
    load_fit,
)

OPTIMIZATION_NUMBER = 16
OPTIMIZATION_NAME = "Pipeline Compute and DMA (3-Stage)"

REQUIRED_PROFILE_FIELDS = ("t_total_us", "t_dma_us", "t_tensor_us")

KNOB_NAMES: dict[str, str] = {
    "n_tiles": (
        "Number of pipeline tiles (M // TILE_M). Pattern requires n_tiles >= 3 "
        "for the 3-stage pipeline to have steady-state savings. Set to null if "
        "the kernel is not tile-loop structured."
    ),
    "has_serial_dma_compute": (
        "True iff the kernel currently serializes DMA load → compute → DMA store "
        "per tile (no software pipeline). Set to false if already pipelined with "
        "ping-pong buffers."
    ),
}


@dataclass
class Knobs:
    n_tiles: int
    has_serial_dma_compute: bool = True


_FIT = load_fit(__file__)


def predict(profile: KernelProfile, knobs: Knobs) -> PredictionDelta | None:
    if not knobs.has_serial_dma_compute or knobs.n_tiles < 3:
        return None
    if profile.t_dma_us is None or profile.t_tensor_us is None:
        return None

    alpha_dma = _FIT["constants"]["alpha_dma"]
    alpha_overhead = _FIT["constants"]["alpha_overhead"]

    t_overhead = profile.t_total_us - profile.t_dma_us - profile.t_tensor_us
    t_after = alpha_dma * profile.t_dma_us + alpha_overhead * t_overhead

    delta_total = t_after - profile.t_total_us
    delta_tensor = -profile.t_tensor_us  # tensor fully hidden behind DMA

    predicted_after = replace(
        profile,
        t_total_us=t_after,
        t_tensor_us=0.0,
        t_active_us=profile.t_dma_us,
    )
    return PredictionDelta(
        predicted_after=predicted_after,
        delta_us={"t_total_us": delta_total, "t_tensor_us": delta_tensor},
        t_after_us=t_after,
        speedup=profile.t_total_us / t_after,
        extrapolation_warning=extrapolation_warning(_FIT, {"n_tiles": knobs.n_tiles}),
        notes=(f"DMA-bound pipeline: hides {profile.t_tensor_us:.2f}us of TensorE "
               f"behind {profile.t_dma_us:.2f}us of DMA"),
    )


if __name__ == "__main__":
    # Self-test: predict against each "before" row in calibration.csv and
    # compare predicted speedup to observed speedup.
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

    print(f"{'kernel_id':>30s}  {'n_tiles':>8s}  {'pred speedup':>13s}  "
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
            t_tensor_us=float(b["t_tensor_us"]),
            t_dma_us=float(b["t_dma_us"]),
        )
        knobs = Knobs(n_tiles=int(b["n_tiles"]), has_serial_dma_compute=True)
        pred = predict(profile, knobs)
        obs_speedup = float(b["t_total_us"]) / float(a["t_total_us"])
        err = abs(pred.speedup - obs_speedup) / obs_speedup * 100
        print(f"{kid:>30s}  {b['n_tiles']:>8s}  {pred.speedup:>12.2f}x  "
              f"{obs_speedup:>11.2f}x  {err:>5.1f}%")
