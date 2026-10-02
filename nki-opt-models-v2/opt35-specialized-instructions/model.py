"""
Performance model for Optimization #35: Specialized Instruction Fusion
======================================================================

Pattern: a kernel manually broadcasts a [P, 1] vector over [P, N] using an
inner ``affine_range(N)`` loop of ``nisa.tensor_tensor`` calls, optionally
followed by an activation. Replacing the manual loop with a fused
``nisa.activation(scale=..., bias=...)`` collapses N inner-loop instruction
dispatches into one Scalar Engine call. Speedups can exceed 500×.

v2 design — caller-anchored
---------------------------
The optimized kernel's runtime is dominated by DMA (unchanged) plus a small
per-outer-iter Scalar Engine dispatch:

    t_after = alpha_dma * t_dma_before
              + alpha_iter * num_outer_iters
              + alpha_fixed

The user's measured ``t_dma_us`` anchors the DMA cost; ``num_outer_iters`` is
a structural knob extracted from kernel source. ``broadcast_dim_N`` and
``fused_ops`` enter only via applicability/extrapolation — they govern how
expensive the *before* loop was, which the user's measured ``t_total_us``
already captures.

Knobs
-----
- ``has_manual_broadcast`` (bool): True iff the kernel uses an
  ``affine_range(N)`` inner loop of ``nisa.tensor_tensor`` to broadcast a
  ``[P, 1]`` tile across ``[P, N]``.
- ``broadcast_dim_N`` (int): inner-loop range (free-axis size). Larger N
  means more inner dispatches eliminated.
- ``fused_ops`` (int): number of element-wise pre-activation ops sharing the
  manual broadcast (e.g., 2 for scale + bias; 1 for scale-only).
- ``num_outer_iters`` (int): how many times the outer loop containing the
  manual broadcast runs.

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

OPTIMIZATION_NUMBER = 35
OPTIMIZATION_NAME = "Specialized Instructions"

REQUIRED_PROFILE_FIELDS = ("t_total_us", "t_dma_us")

KNOB_NAMES: dict[str, str] = {
    "has_manual_broadcast": (
        "True iff the kernel uses an inner affine_range(N) loop of "
        "nisa.tensor_tensor to broadcast a [P, 1] tile across [P, N]. "
        "v3 — ALSO TRUE for any pattern that a specialized NKI ISA "
        "instruction collapses into one dispatch, including: "
        "(a) sqrt+add(eps)+reciprocal chains replaceable by rsqrt, "
        "(b) erf/tanh-based GELU sequences replaceable by nisa.activation "
        "with the gelu opcode, (c) softmax-denom chains "
        "(exp -> tensor_reduce -> reciprocal -> tensor_tensor) replaceable "
        "by nisa.softmax. The speedup mechanism is the same: N inner-loop "
        "instructions collapse into 1 fused ISA call. Set false only if "
        "no specialized instruction matches the kernel's scalar chain."
    ),
    "broadcast_dim_N": (
        "Size of the free axis (N) being manually broadcast in the inner loop "
        "(e.g., the range argument of affine_range(N)). Set to null if the "
        "pattern does not apply."
    ),
    "fused_ops": (
        "Number of element-wise ops (scale, bias) sharing the manual broadcast "
        "loop, fused into one nisa.activation call after the optimization. "
        "Typically 1 (scale-only) or 2 (scale + bias). Null if not applicable."
    ),
    "num_outer_iters": (
        "Number of times the enclosing outer loop runs (e.g., the range "
        "argument of the surrounding affine_range / sequential_range). "
        "Null if not applicable."
    ),
}


@dataclass
class Knobs:
    has_manual_broadcast: bool
    broadcast_dim_N: int
    fused_ops: int
    num_outer_iters: int


_FIT = load_fit(__file__)


def predict(profile: KernelProfile, knobs: Knobs) -> PredictionDelta | None:
    if not knobs.has_manual_broadcast:
        return None
    if knobs.broadcast_dim_N is None or knobs.fused_ops is None:
        return None
    if knobs.num_outer_iters is None or knobs.num_outer_iters < 1:
        return None
    if profile.t_dma_us is None:
        return None

    alpha_dma = _FIT["constants"]["alpha_dma"]
    alpha_iter = _FIT["constants"]["alpha_iter"]
    alpha_fixed = _FIT["constants"]["alpha_fixed"]

    t_after = (alpha_dma * profile.t_dma_us
               + alpha_iter * knobs.num_outer_iters
               + alpha_fixed)
    # Model can predict negative or very small t_after at extreme extrapolation;
    # clamp to a small positive floor (the DMA cost itself).
    t_after = max(t_after, profile.t_dma_us * 0.5)

    delta_total = t_after - profile.t_total_us
    # The before kernel's vector time collapses; predict ~0 after.
    new_vector = 0.0 if profile.t_vector_us is not None else None

    predicted_after = replace(
        profile,
        t_total_us=t_after,
        t_vector_us=new_vector,
        t_active_us=profile.t_dma_us,
    )
    return PredictionDelta(
        predicted_after=predicted_after,
        delta_us={
            "t_total_us": delta_total,
            "t_vector_us": (-(profile.t_vector_us or 0.0)) if profile.t_vector_us is not None else 0.0,
        },
        t_after_us=t_after,
        speedup=profile.t_total_us / t_after if t_after > 0 else float("inf"),
        extrapolation_warning=extrapolation_warning(_FIT, {
            "broadcast_dim_N": knobs.broadcast_dim_N,
            "fused_ops": knobs.fused_ops,
            "num_outer_iters": knobs.num_outer_iters,
        }),
        notes=(f"Eliminates manual broadcast loop: "
               f"{knobs.fused_ops}*{knobs.broadcast_dim_N}*{knobs.num_outer_iters} "
               f"= {knobs.fused_ops * knobs.broadcast_dim_N * knobs.num_outer_iters} "
               f"VectorE dispatches collapse to {knobs.num_outer_iters} fused calls"),
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

    print(f"{'kernel_id':>32s}  {'N':>4s}  {'ops':>3s}  {'iters':>5s}  "
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
            t_vector_us=float(b["t_vector_us"]) if b.get("t_vector_us") else None,
        )
        knobs = Knobs(
            has_manual_broadcast=True,
            broadcast_dim_N=int(b["broadcast_dim_N"]),
            fused_ops=int(b["fused_ops"]),
            num_outer_iters=int(b["num_outer_iters"]),
        )
        pred = predict(profile, knobs)
        obs_speedup = float(b["t_total_us"]) / float(a["t_total_us"])
        err = abs(pred.speedup - obs_speedup) / obs_speedup * 100
        print(f"{kid:>32s}  {b['broadcast_dim_N']:>4s}  {b['fused_ops']:>3s}  "
              f"{b['num_outer_iters']:>5s}  {pred.speedup:>12.2f}x  "
              f"{obs_speedup:>11.2f}x  {err:>5.1f}%")
