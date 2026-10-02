"""
Performance model for Optimization #31: Load Larger Blocks for SBUF Reuse
==========================================================================

Pattern: a kernel tiles the free dimension into small ``TILE_N``-element
chunks (e.g. ``TILE_N=32`` with ``N=512``), then runs an inner loop over the
tiles. Increasing ``TILE_N`` toward ``N`` (a single full-row block) reduces
DMA packet count and VectorE instruction count proportionally, even though
total bytes moved stays the same (the compiler caches HBM either way).

v2 design — caller-anchored
---------------------------
DMA active time and VectorE active time both scale as ``1 / tile_ratio``
(where ``tile_ratio = N / TILE_N_current``); the rest is roughly invariant:

    t_after = alpha_dma    * t_dma_b    / tile_ratio
            + alpha_vector * t_vector_b / tile_ratio
            + alpha_other  * (t_total_b - t_dma_b - t_vector_b)
            + alpha_fixed

The user's measured ``t_dma_us``, ``t_vector_us``, ``t_total_us`` anchor the
scaling laws; ``tile_ratio`` is structural.

Knobs
-----
- ``has_small_tile_pattern`` (bool): True iff the kernel inner-loops over
  ``N // TILE_N`` chunks of the free axis. Set to false (model returns None)
  if the kernel is already single-block or not tile-loop structured.
- ``N`` (int): full free-axis size.
- ``TILE_N_current`` (int): the kernel's current free-axis tile size.

Required profile fields
-----------------------
``t_total_us``, ``t_dma_us``, ``t_vector_us``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from _common import (  # type: ignore[import-not-found]
    KernelProfile,
    PredictionDelta,
    extrapolation_warning,
    load_fit,
)

OPTIMIZATION_NUMBER = 31
OPTIMIZATION_NAME = "Load Larger Blocks for SBUF Reuse"

REQUIRED_PROFILE_FIELDS = ("t_total_us", "t_dma_us", "t_vector_us")

KNOB_NAMES: dict[str, str] = {
    "has_small_tile_pattern": (
        "True iff the kernel tiles the free dimension into small chunks and "
        "inner-loops over them (e.g. `for n in affine_range(N // TILE_N): "
        "load x[..., n*TILE_N:(n+1)*TILE_N]`). Set to false if the kernel "
        "already loads full free-dimension blocks."
    ),
    "tile_body_contains_full_row_reduce": (
        "v3 — REQUIRED for applicability. True iff the inner tile loop body "
        "contains a `tensor_reduce` (or equivalent partial-reduce) over the "
        "tiled axis whose result is later combined across tiles. The opt's "
        "speedup comes from collapsing partial reductions into one full-row "
        "reduce (see nki-optimizations/31-load-larger-blocks/README.md:22-36). "
        "Set false for kernels whose tile loop is a pure element-wise or "
        "store loop — those won't benefit from this optimization, and the "
        "model would silently over-predict on producer-consumer pipelines "
        "(see benchmark-opt/reports/mg_v2_predictor_misses.md: "
        "add_rmsnorm_matmul 15.53x -> 1.19x silent over-prediction)."
    ),
    "is_producer_consumer_pipeline": (
        "v3 — REQUIRED. True iff this tile loop sits in a producer-consumer "
        "pipeline (e.g. matmul output feeding a downstream norm/reduce). "
        "When True, the model returns None because the HBM-caching assumption "
        "(reload same row repeatedly) does not hold — the producer's stores "
        "evict the cache between consumer reads. Set False for standalone "
        "row-wise kernels that re-read the same input across tiles."
    ),
    "N": (
        "Full free-axis size (e.g. the kernel's `N` constant or the full "
        "second dim of the input). Null if not applicable."
    ),
    "TILE_N_current": (
        "The kernel's current free-axis tile size. The optimization replaces "
        "this with TILE_N=N (one block per M-tile). Null if not applicable."
    ),
}


@dataclass
class Knobs:
    has_small_tile_pattern: bool
    N: int
    TILE_N_current: int
    tile_body_contains_full_row_reduce: bool = False
    is_producer_consumer_pipeline: bool = False


_FIT = load_fit(__file__)


def predict(profile: KernelProfile, knobs: Knobs) -> PredictionDelta | None:
    if not knobs.has_small_tile_pattern:
        return None
    # v3: tighten applicability. Without a tile-body reduce (or with a
    # producer-consumer pipeline), the HBM-caching assumption breaks and the
    # model silently over-predicts. See README:22-36 and
    # benchmark-opt/reports/mg_v2_predictor_misses.md.
    if not knobs.tile_body_contains_full_row_reduce:
        return None
    if knobs.is_producer_consumer_pipeline:
        return None
    if knobs.N is None or knobs.TILE_N_current is None:
        return None
    if knobs.TILE_N_current >= knobs.N:
        return None  # already optimal
    if knobs.N % knobs.TILE_N_current != 0:
        return None
    if profile.t_dma_us is None or profile.t_vector_us is None:
        return None

    tile_ratio = knobs.N / knobs.TILE_N_current

    alpha_dma = _FIT["constants"]["alpha_dma"]
    alpha_vector = _FIT["constants"]["alpha_vector"]
    alpha_other = _FIT["constants"]["alpha_other"]
    alpha_fixed = _FIT["constants"]["alpha_fixed"]

    t_other = max(profile.t_total_us - profile.t_dma_us - profile.t_vector_us, 0.0)
    t_after = (alpha_dma * profile.t_dma_us / tile_ratio
               + alpha_vector * profile.t_vector_us / tile_ratio
               + alpha_other * t_other
               + alpha_fixed)
    t_after = max(t_after, profile.t_dma_us / tile_ratio * 0.5)

    delta_total = t_after - profile.t_total_us
    predicted_after = replace(
        profile,
        t_total_us=t_after,
        t_dma_us=profile.t_dma_us / tile_ratio,
        t_vector_us=profile.t_vector_us / tile_ratio,
        t_active_us=profile.t_dma_us / tile_ratio,
    )
    return PredictionDelta(
        predicted_after=predicted_after,
        delta_us={
            "t_total_us": delta_total,
            "t_dma_us": profile.t_dma_us / tile_ratio - profile.t_dma_us,
            "t_vector_us": profile.t_vector_us / tile_ratio - profile.t_vector_us,
        },
        t_after_us=t_after,
        speedup=profile.t_total_us / t_after if t_after > 0 else float("inf"),
        extrapolation_warning=extrapolation_warning(_FIT, {
            "tile_ratio": tile_ratio,
            "N": knobs.N,
        }),
        notes=(f"Increasing TILE_N from {knobs.TILE_N_current} to {knobs.N} "
               f"({tile_ratio:.0f}× fewer DMA packets and VectorE dispatches)"),
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

    print(f"{'kernel_id':>30s}  {'N':>5s}  {'TILE_N':>6s}  {'pred speedup':>13s}  "
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
            t_dma_us=float(b["t_dma_us"]),
            t_vector_us=float(b["t_vector_us"]),
        )
        knobs = Knobs(
            has_small_tile_pattern=True,
            N=int(b["N"]),
            TILE_N_current=int(b["TILE_N_current"]),
            tile_body_contains_full_row_reduce=True,
            is_producer_consumer_pipeline=False,
        )
        pred = predict(profile, knobs)
        obs_speedup = float(b["t_total_us"]) / float(a["t_total_us"])
        err = abs(pred.speedup - obs_speedup) / obs_speedup * 100
        print(f"{kid:>30s}  {b['N']:>5s}  {b['TILE_N_current']:>6s}  "
              f"{pred.speedup:>12.2f}x  {obs_speedup:>11.2f}x  {err:>5.1f}%")
