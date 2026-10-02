"""
Performance model for Optimization #1: SBUF Accumulator
========================================================

Pattern: a reduction kernel keeps a running accumulator in HBM
(``buffer=nl.private_hbm``) and reloads/stores it on every iteration of
``range(M // TILE_M)``. Each iteration pays ~2 unnecessary DMA round-trips
of a ``[TILE_M, N]`` tile. Moving the accumulator to SBUF
(``nl.zeros(..., buffer=nl.sbuf)``) eliminates these round-trips and stores
the result with a single final ``dma_copy``.

v2 design — caller-anchored
---------------------------
The DMA cost shrinks (accumulator round-trips removed) while the compute
cost is preserved. Three-parameter linear caller-anchored model:

    t_after = alpha_dma * t_dma_b + alpha_other * (t_total_b - t_dma_b) + alpha_fixed

We expect ``alpha_dma`` < 1 (most DMA eliminated) and ``alpha_other`` ≈ 1
(VectorE compute preserved). ``M``, ``TILE_M``, ``N`` enter only via
applicability/extrapolation — the caller's measured ``t_total_us`` and
``t_dma_us`` already capture their effect.

Knobs
-----
- ``has_hbm_accumulator`` (bool): True iff the kernel allocates a running
  accumulator in ``private_hbm`` and reloads/stores it inside the loop.
  The profiler signature is ``spill_save_bytes > 0`` paired with
  ``spill_reload_bytes > 0``.
- ``M`` (int): total partition-axis dimension (the dim being reduced).
- ``TILE_M`` (int): partition-tile size driving the loop. ≤ 128.
- ``N`` (int): free-axis size of the accumulator tile (governs spill bytes
  per round-trip).

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

OPTIMIZATION_NUMBER = 1
OPTIMIZATION_NAME = "SBUF Accumulator"

REQUIRED_PROFILE_FIELDS = ("t_total_us", "t_dma_us")

KNOB_NAMES: dict[str, str] = {
    "has_hbm_accumulator": (
        "True iff the kernel allocates a running accumulator in HBM "
        "(buffer=nl.private_hbm) and reloads + stores it on every iteration "
        "of an outer range(M // TILE_M) loop. The profiler signature is "
        "spill_save_bytes > 0 with a per-iteration store+reload pattern. "
        "Set to false if the accumulator is already in SBUF or no such "
        "pattern exists."
    ),
    "M": (
        "Total partition-axis dimension being reduced (rows). The outer "
        "loop is range(M // TILE_M). Null if pattern does not apply."
    ),
    "TILE_M": (
        "Partition tile size driving the reduction loop. Must be ≤ 128. "
        "Null if pattern does not apply."
    ),
    "N": (
        "Free-axis size of the accumulator tile. Governs the bytes spilled "
        "per round-trip (TILE_M * N * dtype_bytes). Null if pattern does "
        "not apply."
    ),
}


@dataclass
class Knobs:
    has_hbm_accumulator: bool
    M: int
    TILE_M: int
    N: int


_FIT = load_fit(__file__)


def predict(profile: KernelProfile, knobs: Knobs) -> PredictionDelta | None:
    if not knobs.has_hbm_accumulator:
        return None
    if knobs.M is None or knobs.TILE_M is None or knobs.N is None:
        return None
    if knobs.TILE_M <= 0 or knobs.M % knobs.TILE_M != 0:
        return None
    if profile.t_dma_us is None:
        return None

    alpha_dma = _FIT["constants"]["alpha_dma"]
    alpha_other = _FIT["constants"]["alpha_other"]
    alpha_fixed = _FIT["constants"]["alpha_fixed"]

    t_other = max(profile.t_total_us - profile.t_dma_us, 0.0)
    t_after = alpha_dma * profile.t_dma_us + alpha_other * t_other + alpha_fixed
    t_after = max(t_after, 1.0)

    delta_total = t_after - profile.t_total_us
    predicted_after = replace(
        profile,
        t_total_us=t_after,
    )
    n_tiles = knobs.M // knobs.TILE_M
    return PredictionDelta(
        predicted_after=predicted_after,
        delta_us={"t_total_us": delta_total},
        t_after_us=t_after,
        speedup=profile.t_total_us / t_after if t_after > 0 else float("inf"),
        extrapolation_warning=extrapolation_warning(_FIT, {
            "M": knobs.M,
            "TILE_M": knobs.TILE_M,
            "N": knobs.N,
        }),
        notes=(f"Hoists accumulator to SBUF: removes {2 * n_tiles} "
               f"unnecessary DMA round-trips of {knobs.TILE_M}x{knobs.N} tile"),
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

    print(f"{'kernel_id':>30s}  {'M':>5s}  {'TILE_M':>6s}  {'N':>5s}  "
          f"{'pred speedup':>13s}  {'obs speedup':>12s}  {'err%':>6s}")
    print("-" * 95)
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
        knobs = Knobs(
            has_hbm_accumulator=True,
            M=int(b["M"]),
            TILE_M=int(b["TILE_M"]),
            N=int(b["N"]),
        )
        pred = predict(profile, knobs)
        obs_speedup = float(b["t_total_us"]) / float(a["t_total_us"])
        err = abs(pred.speedup - obs_speedup) / obs_speedup * 100
        print(f"{kid:>30s}  {b['M']:>5s}  {b['TILE_M']:>6s}  {b['N']:>5s}  "
              f"{pred.speedup:>12.2f}x  {obs_speedup:>11.2f}x  {err:>5.1f}%")
