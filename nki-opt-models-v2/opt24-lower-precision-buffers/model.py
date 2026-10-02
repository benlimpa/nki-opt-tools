"""
Performance model for Optimization #24: Lower-Precision Internal Buffers
========================================================================

Pattern: HBM tensors are stored in ``bfloat16`` but the kernel allocates
internal SBUF tiles as ``float32``. Every ``dma_copy`` triggers an
implicit bf16↔fp32 cast, doubling DMA packet count, and every VectorE op
runs at half throughput. Switching SBUF tiles to ``bfloat16`` removes
the casts, halves SBUF usage, and gives ~2× VectorE throughput.

v2 design — caller-anchored
---------------------------
DMA and VectorE active times each shrink; ScalarE/GPSIMD/idle stays.
Four-parameter linear caller-anchored model:

    t_other = t_total_b - t_dma_b - t_vector_b
    t_after = alpha_dma * t_dma_b
            + alpha_vector * t_vector_b
            + alpha_other * t_other
            + alpha_fixed

Empirically (5-point sweep, P=64–128, N=16K–131K): alpha_dma ≈ 0.7
(implicit cast removal helps, but DMA is still ~70% of fp32),
alpha_vector ≈ 0.58 (half-precision VectorE throughput), alpha_other ≈ 1
(scalar engine + idle preserved).

Knobs
-----
- ``has_fp32_internal_buffers`` (bool): True iff the kernel allocates SBUF
  tiles as ``nl.float32`` while HBM tensors are ``bfloat16``/``float16``
  (every dma_copy implicitly casts). Set to false if the SBUF dtype
  already matches the HBM dtype.
- ``P`` (int): partition-axis size (typically 128).
- ``N`` (int): free-axis total size of the input.
- ``n_fp32_intermediates`` (int): how many fp32 SBUF allocations exist
  per tile iteration (governs the magnitude of the cast/throughput
  savings). Used for extrapolation only.

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

OPTIMIZATION_NUMBER = 24
OPTIMIZATION_NAME = "Lower-Precision Internal Buffers"

REQUIRED_PROFILE_FIELDS = ("t_total_us", "t_dma_us", "t_vector_us")

KNOB_NAMES: dict[str, str] = {
    "has_fp32_internal_buffers": (
        "True iff the kernel allocates internal SBUF tiles as nl.float32 "
        "while the HBM input/output tensors are bfloat16 (or float16). "
        "Each dma_copy then implicitly casts and doubles DMA packets. "
        "Set to false if the SBUF dtype already matches the HBM dtype."
    ),
    "P": (
        "Partition-axis size of the SBUF tile (typically 128 = NeuronCore "
        "partition count). Null if not applicable."
    ),
    "N": (
        "Free-axis total size of the input tensor. Null if not applicable."
    ),
    "n_fp32_intermediates": (
        "Number of fp32 SBUF allocations per tile iteration that will be "
        "downcast to bf16. Higher counts amplify the savings. Null if "
        "not applicable."
    ),
}


@dataclass
class Knobs:
    has_fp32_internal_buffers: bool
    P: int
    N: int
    n_fp32_intermediates: int


_FIT = load_fit(__file__)


def predict(profile: KernelProfile, knobs: Knobs) -> PredictionDelta | None:
    if not knobs.has_fp32_internal_buffers:
        return None
    if knobs.P is None or knobs.N is None:
        return None
    if profile.t_dma_us is None or profile.t_vector_us is None:
        return None

    alpha_dma = _FIT["constants"]["alpha_dma"]
    alpha_vector = _FIT["constants"]["alpha_vector"]
    alpha_other = _FIT["constants"]["alpha_other"]
    alpha_fixed = _FIT["constants"]["alpha_fixed"]

    t_other = max(profile.t_total_us - profile.t_dma_us - profile.t_vector_us, 0.0)
    t_after = (alpha_dma * profile.t_dma_us
               + alpha_vector * profile.t_vector_us
               + alpha_other * t_other
               + alpha_fixed)
    t_after = max(t_after, 1.0)

    delta_total = t_after - profile.t_total_us
    new_vector = profile.t_vector_us * alpha_vector
    predicted_after = replace(
        profile,
        t_total_us=t_after,
        t_vector_us=new_vector,
        t_dma_us=profile.t_dma_us * alpha_dma,
    )
    return PredictionDelta(
        predicted_after=predicted_after,
        delta_us={
            "t_total_us": delta_total,
            "t_vector_us": new_vector - profile.t_vector_us,
            "t_dma_us": profile.t_dma_us * (alpha_dma - 1.0),
        },
        t_after_us=t_after,
        speedup=profile.t_total_us / t_after if t_after > 0 else float("inf"),
        extrapolation_warning=extrapolation_warning(_FIT, {
            "P": knobs.P,
            "N": knobs.N,
            "n_fp32_intermediates": knobs.n_fp32_intermediates,
        }),
        notes=(f"Downcasts {knobs.n_fp32_intermediates} fp32 SBUF buffers "
               f"to bf16: removes implicit DMA casts, ~2x VectorE throughput"),
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

    print(f"{'kernel_id':>30s}  {'P':>4s}  {'N':>7s}  "
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
            t_vector_us=float(b["t_vector_us"]),
        )
        knobs = Knobs(
            has_fp32_internal_buffers=True,
            P=int(b["P"]),
            N=int(b["N"]),
            n_fp32_intermediates=int(b["n_fp32_intermediates"]),
        )
        pred = predict(profile, knobs)
        obs_speedup = float(b["t_total_us"]) / float(a["t_total_us"])
        err = abs(pred.speedup - obs_speedup) / obs_speedup * 100
        print(f"{kid:>30s}  {b['P']:>4s}  {b['N']:>7s}  "
              f"{pred.speedup:>12.2f}x  {obs_speedup:>11.2f}x  {err:>5.1f}%")
