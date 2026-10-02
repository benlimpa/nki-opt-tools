"""
Performance model for Optimization #32: Multi-level Tiling
==========================================================

Pattern: a matmul kernel iterates ``affine_range`` over N-tiles in
the outer loop and K-tiles in the inner loop, reloading the
stationary operand A[k, m] once *per N-tile iteration*. For large
N, A bytes get loaded ``n_n_tiles`` times. Multi-level tiling moves
K to the outer level, loading each A[k] tile once and reusing it
across all N-tiles via a wider PSUM accumulator.

v2 design — caller-anchored
---------------------------
The savings are bandwidth-bound: extra A loads in the suboptimal
order. Compute the structural DMA ratio from the matmul shape:

    a_extra_bytes = M · K · 4 · (n_n_tiles − 1)   # extra A loads
    total_dma_b   = read_bytes + write_bytes (anchored to t_dma_b)
    r_dma_after   = (total_dma_b_bytes − a_extra_bytes) / total_dma_b_bytes

    t_after = α_dma · t_dma_b · r_dma_after
            + α_other · t_total_b
            + α_fixed

Three free parameters; fit by 3x3 LSQ. We anchor the "savings" to
``t_dma_b · (1 − r_dma_after)`` and let the regression weight it.
``t_other_b`` overlaps DMA inside t_total, so we re-anchor on the
full ``t_total_b`` rather than the (often-negative) residual.

Knobs
-----
- ``has_n_outer_k_inner_with_a_reload`` (bool): True iff the
  matmul's M-tile loop is N-outer / K-inner with A loaded in the
  inner K loop (so A reloads per N-tile). Profiler signature:
  ``hbm_read_bytes`` is much greater than ``M·K·4 + K·N·4`` (would
  be optimal). Set false if the compiler already hoists A loads or
  the loop order is already K-outer.
- ``M``, ``K``, ``N`` (int): matmul output shape (lhsT.T @ rhs).
- ``n_n_tiles`` (int): number of N-tiles (= N // TILE_N). Each
  N-tile triggers a full A reload in the suboptimal order.

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

OPTIMIZATION_NUMBER = 32
OPTIMIZATION_NAME = "Multi-level Tiling (Stationary Operand Reuse)"

REQUIRED_PROFILE_FIELDS = ("t_total_us", "t_dma_us")

KNOB_NAMES: dict[str, str] = {
    "has_n_outer_k_inner_with_a_reload": (
        "True iff the matmul kernel iterates N-tiles in the OUTER "
        "loop and K-tiles in the INNER loop, with A (lhsT) loaded "
        "inside the K loop (so A reloads once per N-tile). Profiler "
        "signature: hbm_read_bytes > (M·K + K·N)·4. Set false if the "
        "compiler already hoists A loads or the loop order is K-outer."
    ),
    "M": "Matmul output rows (lhsT.T @ rhs → [M, N]).",
    "K": "Matmul reduction dimension.",
    "N": "Matmul output columns.",
    "n_n_tiles": (
        "Number of N-tiles in the outer loop (= N // TILE_N). Each "
        "N-tile currently triggers a full A reload."
    ),
}


@dataclass
class Knobs:
    has_n_outer_k_inner_with_a_reload: bool
    M: int
    K: int
    N: int
    n_n_tiles: int


_FIT = load_fit(__file__)


def predict(profile: KernelProfile, knobs: Knobs) -> PredictionDelta | None:
    if not knobs.has_n_outer_k_inner_with_a_reload:
        return None
    if profile.t_dma_us is None:
        return None
    if (knobs.M is None or knobs.K is None or knobs.N is None
            or knobs.n_n_tiles is None):
        return None
    if knobs.n_n_tiles < 2:
        return None  # no extra reloads to eliminate

    # Structural read-byte savings:
    #   read_before = M·K·n_n_tiles·4 + K·N·4  (A reloaded per N-tile)
    #   read_after  = M·K·4 + K·N·4
    #   write_bytes = M·N·4 (unchanged)
    a_bytes = knobs.M * knobs.K * 4
    b_bytes = knobs.K * knobs.N * 4
    write_bytes = knobs.M * knobs.N * 4
    read_before = a_bytes * knobs.n_n_tiles + b_bytes
    read_after = a_bytes + b_bytes
    total_before = read_before + write_bytes
    total_after = read_after + write_bytes
    r_dma_after = total_after / total_before if total_before > 0 else 1.0

    alpha_dma = _FIT["constants"]["alpha_dma"]
    alpha_other = _FIT["constants"]["alpha_other"]
    alpha_fixed = _FIT["constants"]["alpha_fixed"]

    t_after = (alpha_dma * profile.t_dma_us * r_dma_after
               + alpha_other * profile.t_total_us
               + alpha_fixed)
    t_after = max(t_after, 1.0)

    delta_total = t_after - profile.t_total_us
    predicted_after = replace(profile, t_total_us=t_after)
    # v3: M alone being out of range should NOT trigger an extrapolation
    # warning that deprioritizes opt32 — the structural ratio r_dma_after
    # already captures the M-dependence, and the cost model is largely
    # M-invariant when M is large. Strip the M field from the extrapolation
    # check; surface the actual M value in `notes` so the agent can still
    # scrutinise. See mg_v2_predictor_misses.md case A2 (nkibench/matmul,
    # M=4096 deprioritised opt32 unnecessarily).
    warn = extrapolation_warning(_FIT, {
        "K": knobs.K, "N": knobs.N,
        "n_n_tiles": knobs.n_n_tiles,
    })
    extra_note = ""
    if knobs.M is not None:
        m_range = _FIT.get("knob_range", {}).get("M")
        if m_range is not None and (knobs.M < m_range[0] or knobs.M > m_range[1]):
            extra_note = (f" [M={knobs.M} is outside calibrated range "
                          f"{m_range} — model is M-invariant in r_dma_after, "
                          f"but verify empirically]")
    return PredictionDelta(
        predicted_after=predicted_after,
        delta_us={"t_total_us": delta_total},
        t_after_us=t_after,
        speedup=profile.t_total_us / t_after if t_after > 0 else float("inf"),
        extrapolation_warning=warn,
        notes=(f"K-outer / N-inner reordering eliminates "
               f"{(knobs.n_n_tiles - 1) * knobs.M * knobs.K * 4 / 1e6:.1f} MB of "
               f"redundant A-tile reads (r_dma_after={r_dma_after:.3f})"
               f"{extra_note}"),
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

    print(f"{'kernel_id':>30s}  {'M':>5s} {'K':>5s} {'N':>5s} {'n_n':>4s}  "
          f"{'pred speedup':>13s}  {'obs speedup':>12s}  {'err%':>6s}")
    print("-" * 100)
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
            has_n_outer_k_inner_with_a_reload=True,
            M=int(b["M"]), K=int(b["K"]), N=int(b["N"]),
            n_n_tiles=int(b["n_n_tiles"]),
        )
        pred = predict(profile, knobs)
        obs_speedup = float(b["t_total_us"]) / float(a["t_total_us"])
        err = abs(pred.speedup - obs_speedup) / obs_speedup * 100
        print(f"{kid:>30s}  {b['M']:>5s} {b['K']:>5s} {b['N']:>5s} "
              f"{b['n_n_tiles']:>4s}  "
              f"{pred.speedup:>12.2f}x  {obs_speedup:>11.2f}x  {err:>5.1f}%")
