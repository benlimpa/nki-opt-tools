"""
Performance model for Optimization #33: Coalesce SBUF Tiles for HBM Store
=========================================================================

Pattern: a kernel issues multiple separate ``nisa.dma_copy`` calls
covering small, adjacent SBUF→HBM ranges. Each call is a distinct
DMA packet — per-packet bus initiation and descriptor processing
overhead adds up. Coalescing: allocate a wider SBUF tile, copy the
small tiles into it via SBUF→SBUF dma_copy, then issue a single
wide SBUF→HBM transfer. SBUF write bytes go up; HBM packet count
goes down.

v2 design — caller-anchored
---------------------------
Anchor the savings to the *number* of fragmented store packets that
get eliminated, not bytes:

    extra_packets = (n_fragments_per_iter − 1) · n_iters

    t_after = α_total · t_total_b
            + α_packet · extra_packets         # fits negative — savings
            + α_fixed

Three free parameters; fit by 3x3 LSQ. ``α_packet`` represents the
per-packet net cost (HBM bus init minus SRAM-copy overhead).

Knobs
-----
- ``has_fragmented_stores`` (bool): True iff the kernel issues
  multiple ``nisa.dma_copy(... → HBM)`` calls to *adjacent* HBM
  addresses inside the same iteration. Profiler signature: high
  ``dma_packet_count`` relative to total HBM write bytes (each
  packet is a small slice of a larger row). Set false if stores
  are already wide / coalesced.
- ``n_iters`` (int): outer loop iteration count.
- ``n_fragments_per_iter`` (int): number of separate SBUF→HBM
  stores per iteration that will be merged into one (≥ 2).

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

OPTIMIZATION_NUMBER = 33
OPTIMIZATION_NAME = "Coalesce SBUF Tiles for HBM Store"

REQUIRED_PROFILE_FIELDS = ("t_total_us",)

KNOB_NAMES: dict[str, str] = {
    "has_fragmented_stores": (
        "True iff the kernel issues multiple nisa.dma_copy(...→HBM) "
        "calls to *adjacent* HBM addresses inside the same iteration. "
        "Profiler signature: high dma_packet_count relative to total "
        "HBM write bytes. Set false if stores are already wide / "
        "coalesced via a single dma_copy of a wide SBUF tile."
    ),
    "n_iters": (
        "Outer loop iteration count of the kernel section that "
        "performs the fragmented stores."
    ),
    "n_fragments_per_iter": (
        "Number of separate SBUF→HBM dma_copy calls per iteration "
        "that will be merged into one. Must be ≥ 2."
    ),
}


@dataclass
class Knobs:
    has_fragmented_stores: bool
    n_iters: int
    n_fragments_per_iter: int


_FIT = load_fit(__file__)


def infer_applicable_from_profile(
    profile: KernelProfile, opt_features: dict
) -> dict | None:
    """v3 profile-driven applicability override.

    Profile signature for store fragmentation: many DMA packets relative to
    HBM write bytes — the kernel emits lots of small stores instead of a few
    wide ones. Heuristic: average HBM-write-side packet size below 16 KiB AND
    dma_packet_count >= 16. The agent must still supply ``n_iters`` and
    ``n_fragments_per_iter`` for the savings calculation; if either is
    missing we defer to the agent.

    See benchmark-opt/reports/mg_v2_predictor_misses.md case A2: opt33 not
    ranked on nkibench/matmul despite ~28x HBM over-fetch (9.5 GB vs 336 MB
    logical inputs).
    """
    if (profile.dma_packet_count is None or profile.dma_packet_count < 16
            or not profile.hbm_write_bytes):
        return None
    avg_pkt = profile.hbm_write_bytes / profile.dma_packet_count
    if avg_pkt >= 16 * 1024:  # >= 16 KiB average packet → already coalesced
        return None
    n_iters = opt_features.get("n_iters")
    n_fragments = opt_features.get("n_fragments_per_iter")
    if not n_iters or not n_fragments or n_fragments < 2:
        return None
    return {
        "has_fragmented_stores": True,
        "n_iters": n_iters,
        "n_fragments_per_iter": n_fragments,
    }


def predict(profile: KernelProfile, knobs: Knobs) -> PredictionDelta | None:
    if not knobs.has_fragmented_stores:
        return None
    if knobs.n_iters is None or knobs.n_iters < 1:
        return None
    if knobs.n_fragments_per_iter is None or knobs.n_fragments_per_iter < 2:
        return None

    alpha_total = _FIT["constants"]["alpha_total"]
    alpha_packet = _FIT["constants"]["alpha_packet"]
    alpha_fixed = _FIT["constants"]["alpha_fixed"]

    extra_packets = (knobs.n_fragments_per_iter - 1) * knobs.n_iters
    t_after = (alpha_total * profile.t_total_us
               + alpha_packet * extra_packets
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
            "n_iters": knobs.n_iters,
            "n_fragments_per_iter": knobs.n_fragments_per_iter,
        }),
        notes=(f"Coalesces {knobs.n_fragments_per_iter} per-iter stores into 1, "
               f"eliminating {extra_packets} HBM-bound DMA packets"),
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

    print(f"{'kernel_id':>30s}  {'iters':>5s}  {'frag':>4s}  "
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
        )
        knobs = Knobs(
            has_fragmented_stores=True,
            n_iters=int(b["n_iters"]),
            n_fragments_per_iter=int(b["n_fragments_per_iter"]),
        )
        pred = predict(profile, knobs)
        obs_speedup = float(b["t_total_us"]) / float(a["t_total_us"])
        err = abs(pred.speedup - obs_speedup) / obs_speedup * 100
        print(f"{kid:>30s}  {b['n_iters']:>5s}  {b['n_fragments_per_iter']:>4s}  "
              f"{pred.speedup:>12.2f}x  {obs_speedup:>11.2f}x  {err:>5.1f}%")
