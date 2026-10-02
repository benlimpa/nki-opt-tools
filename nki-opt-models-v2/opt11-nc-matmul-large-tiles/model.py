"""
Performance model for Optimization #11: nc_matmul on Large Contiguous Blocks
=============================================================================

Pattern: a matmul kernel calls ``nisa.nc_matmul`` with a moving tile that is
narrower than the hardware maximum (``gemm_moving_fmax = 512``). Each
M-tile then issues ``ceil(N / TILE_N_current)`` separate matmul instructions
plus matching PSUM evictions and HBM stores. Widening the moving tile to
``min(N, 512)`` collapses these into a single call per M-tile, cutting
TensorE instructions, PSUM evictions, and result-store transactions.

v2 design — caller-anchored
---------------------------
The TensorE active time and per-call DMA overhead both shrink by the
structural ratio ``r = TILE_N_current / min(N, TILE_N_MAX)``. The bulk
data movement (lhsT loads, rhs bytes, result write bytes) is unchanged.

    t_after = alpha_iter * t_total_b * r + alpha_overhead * (1 - r)

Two-parameter overhead-aware caller-anchored model (same shape as opt03).
Fits via 2x2 LSQ.

Knobs
-----
- ``has_small_nc_matmul_tiles`` (bool): True iff the kernel calls
  ``nisa.nc_matmul`` with a moving tile width strictly less than
  ``min(N, 512)``. Set to false if already at maximum.
- ``M``, ``K``, ``N`` (int): the matmul shape. Used for applicability.
- ``TILE_N_current`` (int): current moving-tile width (the value of
  the inner-N loop step). Must be < min(N, 512).

Required profile fields
-----------------------
``t_total_us``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from _common import (  # type: ignore[import-not-found]
    KernelProfile,
    PredictionDelta,
    extrapolation_warning,
    load_fit,
)

OPTIMIZATION_NUMBER = 11
OPTIMIZATION_NAME = "nc_matmul on Large Contiguous Blocks"

REQUIRED_PROFILE_FIELDS = ("t_total_us",)

TILE_N_MAX: int = 512  # gemm_moving_fmax

KNOB_NAMES: dict[str, str] = {
    "has_small_nc_matmul_tiles": (
        "True iff the kernel calls nisa.nc_matmul with a moving-tile width "
        "(the inner N-loop step / second dim of the moving operand) "
        "strictly less than min(N, 512). Set to false if already at the "
        "hardware maximum."
    ),
    "M": "Matmul output rows (= stationary operand free dim). Null if not applicable.",
    "K": "Matmul contraction dim (= partition dim of both operands). Null if not applicable.",
    "N": "Matmul output cols (= moving operand free dim). Null if not applicable.",
    "TILE_N_current": (
        "Current moving-tile width (the value of the inner N-loop step). "
        "Must be < min(N, 512) to be optimizable. Null if not applicable."
    ),
}


@dataclass
class Knobs:
    has_small_nc_matmul_tiles: bool
    M: int
    K: int
    N: int
    TILE_N_current: int


_FIT = load_fit(__file__)


def predict(profile: KernelProfile, knobs: Knobs) -> PredictionDelta | None:
    if not knobs.has_small_nc_matmul_tiles:
        return None
    if knobs.N is None or knobs.TILE_N_current is None:
        return None
    tile_n_target = min(knobs.N, TILE_N_MAX)
    if knobs.TILE_N_current >= tile_n_target:
        return None
    if profile.t_total_us is None:
        return None

    alpha_iter = _FIT["constants"]["alpha_iter"]
    alpha_overhead = _FIT["constants"]["alpha_overhead"]

    r = knobs.TILE_N_current / tile_n_target
    t_after = alpha_iter * profile.t_total_us * r + alpha_overhead * (1.0 - r)
    t_after = max(t_after, 1.0)

    delta_total = t_after - profile.t_total_us
    predicted_after = replace(
        profile,
        t_total_us=t_after,
    )
    n_before = -(-knobs.N // knobs.TILE_N_current)  # ceil
    n_after = -(-knobs.N // tile_n_target)
    return PredictionDelta(
        predicted_after=predicted_after,
        delta_us={"t_total_us": delta_total},
        t_after_us=t_after,
        speedup=profile.t_total_us / t_after if t_after > 0 else float("inf"),
        extrapolation_warning=extrapolation_warning(_FIT, {
            "M": knobs.M,
            "K": knobs.K,
            "N": knobs.N,
            "TILE_N_current": knobs.TILE_N_current,
        }),
        notes=(f"Widens nc_matmul moving tile {knobs.TILE_N_current}→"
               f"{tile_n_target}: collapses {n_before} calls per M-tile "
               f"into {n_after}"),
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

    print(f"{'kernel_id':>30s}  {'M':>5s}  {'N':>5s}  {'TILE_N':>6s}  "
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
        )
        knobs = Knobs(
            has_small_nc_matmul_tiles=True,
            M=int(b["M"]),
            K=int(b["K"]),
            N=int(b["N"]),
            TILE_N_current=int(b["TILE_N_current"]),
        )
        pred = predict(profile, knobs)
        obs_speedup = float(b["t_total_us"]) / float(a["t_total_us"])
        err = abs(pred.speedup - obs_speedup) / obs_speedup * 100
        print(f"{kid:>30s}  {b['M']:>5s}  {b['N']:>5s}  {b['TILE_N_current']:>6s}  "
              f"{pred.speedup:>12.2f}x  {obs_speedup:>11.2f}x  {err:>5.1f}%")
