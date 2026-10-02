"""
Performance model for Optimization #14: Double Buffering
=========================================================

Pattern: a matmul kernel iterates ``sequential_range`` over M-tiles
with a single SBUF buffer for the input operand (e.g., lhsT). DMA
and TensorE serialize because each ``nc_matmul`` has a data
dependency on the just-loaded buffer.

Switching to two alternating SBUF slots lets the DMA load for tile
m+1 issue *concurrently* with ``nc_matmul`` on tile m. (n_tiles − 1)
of the n_tiles iterations then run with DMA hidden behind compute.

v2 design — caller-anchored
---------------------------
The savings scale with the *minimum* of measured DMA and TensorE
times (whichever is the shorter side gets fully hidden in each
overlapped iteration), discounted by the prologue/epilogue:

    overlap_term = (n_tiles − 1) / n_tiles · min(t_dma_b, t_tensor_b)

    t_after = α_total · t_total_b
            + α_overlap · overlap_term       # fits negative — savings
            + α_fixed

Three free parameters; fit by 3x3 LSQ. ``α_overlap`` will fit
negative; magnitude ≈ 1.0 corresponds to perfect overlap, calibration
typically lands around −0.7 to −0.9 (η_overlap ≈ 0.7–0.9).

Knobs
-----
- ``has_serial_dma_compute`` (bool): True iff the matmul iterates
  ``sequential_range`` (or any serial-dep loop) over M-tiles with a
  single SBUF buffer for lhsT (or equivalent loop-carried operand).
  Profiler signature: ``software_dynamic_dma_active_time`` is large
  relative to TensorE active time AND ``tensor_engine_active_time``
  is bounded by serial DMA. Set false if the kernel already
  double-buffers or has no per-tile DMA.
- ``n_tiles`` (int): number of sequential_range iterations to
  double-buffer (M // TILE_M). Must be ≥ 2.

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

OPTIMIZATION_NUMBER = 14
OPTIMIZATION_NAME = "Double Buffering"

REQUIRED_PROFILE_FIELDS = ("t_total_us", "t_dma_us", "t_tensor_us")

KNOB_NAMES: dict[str, str] = {
    "has_serial_dma_compute": (
        "True iff a sequential_range (or otherwise serial-dep) loop "
        "over M-tiles uses a single SBUF buffer for the loop-carried "
        "operand (e.g. lhsT). Profiler signature: per-tile DMA on the "
        "loop-carried operand is comparable to per-tile TensorE time, "
        "and they execute serially. Set false if the kernel already "
        "double-buffers, has no per-tile DMA, or n_tiles < 2."
    ),
    "loop_carried_operand_is_per_tile_loaded": (
        "v3 — REQUIRED. True iff the matmul kernel has a single SBUF "
        "buffer for an operand that is RE-LOADED FROM HBM every tile "
        "(the canonical double-buffering target). False if the loop-"
        "carried operand is computed in SBUF (no DMA to overlap), or "
        "if DMA cost is already dominated by writes rather than the "
        "per-tile read. Without this, the model silently over-predicts "
        "on layernorm-style kernels where the per-tile DMA is store "
        "traffic (see benchmark-opt/reports/presentation_slides.md: "
        "opt14 on nl_to_nki/layernorm 3.22x -> 1.00x silent error)."
    ),
    "n_tiles": (
        "Number of sequential_range iterations the optimization will "
        "double-buffer (typically M // TILE_M). Must be ≥ 2."
    ),
}


@dataclass
class Knobs:
    has_serial_dma_compute: bool
    n_tiles: int
    loop_carried_operand_is_per_tile_loaded: bool = False


_FIT = load_fit(__file__)


def predict(profile: KernelProfile, knobs: Knobs) -> PredictionDelta | None:
    if not knobs.has_serial_dma_compute:
        return None
    if not knobs.loop_carried_operand_is_per_tile_loaded:
        return None
    if knobs.n_tiles is None or knobs.n_tiles < 2:
        return None
    if profile.t_dma_us is None or profile.t_tensor_us is None:
        return None

    alpha_total = _FIT["constants"]["alpha_total"]
    alpha_overlap = _FIT["constants"]["alpha_overlap"]
    alpha_fixed = _FIT["constants"]["alpha_fixed"]

    overlap_term = ((knobs.n_tiles - 1) / knobs.n_tiles
                    * min(profile.t_dma_us, profile.t_tensor_us))
    t_after = (alpha_total * profile.t_total_us
               + alpha_overlap * overlap_term
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
        }),
        notes=(f"Double-buffers {knobs.n_tiles} M-tiles; overlapping "
               f"min(DMA, TensorE)·(n−1)/n ≈ {overlap_term:.2f} µs"),
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

    print(f"{'kernel_id':>30s}  {'n_tiles':>7s}  "
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
            t_dma_us=float(b["t_dma_us"]),
            t_tensor_us=float(b["t_tensor_us"]),
        )
        knobs = Knobs(
            has_serial_dma_compute=True,
            n_tiles=int(b["n_tiles"]),
            loop_carried_operand_is_per_tile_loaded=True,
        )
        pred = predict(profile, knobs)
        obs_speedup = float(b["t_total_us"]) / float(a["t_total_us"])
        err = abs(pred.speedup - obs_speedup) / obs_speedup * 100
        print(f"{kid:>30s}  {b['n_tiles']:>7s}  "
              f"{pred.speedup:>12.2f}x  {obs_speedup:>11.2f}x  {err:>5.1f}%")
