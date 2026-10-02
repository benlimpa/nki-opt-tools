"""
Performance model for Optimization #13: Lower Input Precision (bfloat16
matmul inputs)
=======================================================================

Pattern: a matmul kernel loads input tiles as ``float32`` (4 B/elem).
Switching to ``bfloat16`` (2 B/elem) halves input HBM read bytes, and
``nc_matmul`` accepts bf16 stationary/moving with f32 PSUM accumulation —
output precision unchanged. Two effects:

1. **DMA savings** proportional to the read-byte fraction of total DMA.
   Writes stay f32 and are unchanged.
2. **TensorE savings** — the compiler emits fewer/wider TensorE
   instructions for bf16 input tiles (~0.4× active time observed).

v2 design — caller-anchored
---------------------------
Decompose the user's measured profile into DMA + TensorE, then apply
structural ratios. (DMA and TensorE active times overlap inside t_total
so we can't introduce a clean "other" term — the fixed offset absorbs
the small scalar/vector residual.)

    read_bytes  = (M·K + K·N) · 4               # f32 input reads
    write_bytes = M·N · 4                       # f32 output write (unchanged)
    r_dma = (read_bytes·dtype_ratio + write_bytes) / (read_bytes + write_bytes)

    t_after = α_dma   · t_dma_b   · r_dma
            + α_tensor · t_tensor_b
            + α_fixed

Three free parameters; fit by 3x3 LSQ. ``α_tensor`` absorbs the bf16
TensorE speedup empirically (no need to bake a target ratio in).

Knobs
-----
- ``has_fp32_inputs`` (bool): True iff the matmul currently loads inputs
  as f32 in HBM. Profiler signature: ``hbm_read_bytes`` matches
  ``(M·K + K·N) · 4`` (vs `· 2` for bf16). Set false if inputs are
  already bf16/fp16.
- ``M``, ``K``, ``N`` (int): matmul output shape. Used to compute the
  structural DMA ratio ``r_dma``.
- ``dtype_ratio`` (float): target_bytes / current_bytes (0.5 for
  bf16←f32, 0.25 for int8←f32). Default 0.5.

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

OPTIMIZATION_NUMBER = 13
OPTIMIZATION_NAME = "Lower Input Precision (bfloat16 matmul inputs)"

REQUIRED_PROFILE_FIELDS = ("t_total_us", "t_dma_us", "t_tensor_us")

KNOB_NAMES: dict[str, str] = {
    "has_fp32_inputs": (
        "True iff the matmul currently loads input tiles (lhsT, rhs) "
        "as float32 in HBM. Profiler signature: hbm_read_bytes matches "
        "(M·K + K·N)·4. Set false if inputs are already bfloat16/fp16. "
        "Output precision is unchanged (PSUM accumulates in f32 regardless)."
    ),
    "M": "Matmul output rows (lhsT.T @ rhs → [M, N]).",
    "K": "Matmul reduction dimension.",
    "N": "Matmul output columns.",
    "dtype_ratio": (
        "Ratio target_bytes/current_bytes for the input dtype switch. "
        "0.5 for bfloat16 from float32 (default). 0.25 for int8 from float32. "
        "Set null to use the default 0.5."
    ),
}


@dataclass
class Knobs:
    has_fp32_inputs: bool
    M: int
    K: int
    N: int
    dtype_ratio: float | None = 0.5


_FIT = load_fit(__file__)


def predict(profile: KernelProfile, knobs: Knobs) -> PredictionDelta | None:
    if not knobs.has_fp32_inputs:
        return None
    if profile.t_dma_us is None or profile.t_tensor_us is None:
        return None
    if knobs.M is None or knobs.K is None or knobs.N is None:
        return None
    if knobs.M <= 0 or knobs.K <= 0 or knobs.N <= 0:
        return None

    dtype_ratio = knobs.dtype_ratio if knobs.dtype_ratio is not None else 0.5
    if not 0 < dtype_ratio < 1:
        return None

    read_bytes = (knobs.M * knobs.K + knobs.K * knobs.N) * 4
    write_bytes = knobs.M * knobs.N * 4
    total_dma = read_bytes + write_bytes
    r_dma = (read_bytes * dtype_ratio + write_bytes) / total_dma

    alpha_dma = _FIT["constants"]["alpha_dma"]
    alpha_tensor = _FIT["constants"]["alpha_tensor"]
    alpha_fixed = _FIT["constants"]["alpha_fixed"]

    # Note: TensorE and DMA active times overlap inside t_total, so we can't
    # decompose with a separate t_other term. The fixed offset absorbs the
    # non-DMA, non-TensorE residual (scalar/vector/etc.), which is small.
    t_after = (alpha_dma * profile.t_dma_us * r_dma
               + alpha_tensor * profile.t_tensor_us
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
            "M": knobs.M, "K": knobs.K, "N": knobs.N,
        }),
        notes=(f"Switches inputs from f32 to bf16 (dtype_ratio={dtype_ratio:.2f}); "
               f"DMA ratio r_dma={r_dma:.3f}, TensorE scaled by α_tensor"),
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

    print(f"{'kernel_id':>30s}  {'M':>5s} {'K':>5s} {'N':>5s}  "
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
            has_fp32_inputs=True,
            M=int(b["M"]), K=int(b["K"]), N=int(b["N"]),
            dtype_ratio=0.5,
        )
        pred = predict(profile, knobs)
        obs_speedup = float(b["t_total_us"]) / float(a["t_total_us"])
        err = abs(pred.speedup - obs_speedup) / obs_speedup * 100
        print(f"{kid:>30s}  {b['M']:>5s} {b['K']:>5s} {b['N']:>5s}  "
              f"{pred.speedup:>12.2f}x  {obs_speedup:>11.2f}x  {err:>5.1f}%")
