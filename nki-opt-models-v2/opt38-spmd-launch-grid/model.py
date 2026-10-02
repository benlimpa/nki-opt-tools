"""
Performance model for Optimization #38: SPMD Launch Grid
=========================================================

Pattern: a kernel currently launched as ``kernel(args)`` (no SPMD grid, so
``nl.num_programs(0) == 1``) whose outer loop iterations are independent
(no cross-iteration scan/accumulator/carry). The fix: launch the same kernel
as ``kernel[G](args)`` and shard the outer-loop iteration range across the
G program instances using ``nl.program_id(0)`` / ``nl.num_programs(0)``.

Each instance maps to a logical NeuronCore. The runtime maps logical cores
to physical NCs (1 on Trainium1, 2 on Trainium2). The agent should sweep
``target_grid_size`` candidates (typically 1, 2) and pick the highest
predicted speedup that does not warn for extrapolation.

v2 design — caller-anchored
---------------------------
With G = ``target_grid_size`` and ``r = 1 / G`` (= n_after / n_before for the
parallelisable iteration count), the cost decomposes as a parallelisable share
that scales with r and an irreducible overhead that does not:

    t_after = alpha_par * t_total_b * r + alpha_overhead * (1 - r)

Two free parameters fit via 2x2 LSQ over paired before/after observations
where the after kernel was launched at G > 1. The model degrades gracefully
when G = 1 (returns 1× speedup, identical to before).

Knobs
-----
- ``has_no_spmd_grid`` (bool): True iff the wrapper currently invokes the
  kernel as ``kernel(...)`` (no launch-grid syntax). Set False if the kernel
  is already launched as ``kernel[G](...)`` with G > 1.
- ``has_independent_outer_loop`` (bool): True iff the outer ``sequential_range``
  iterations are independent — no running scan, accumulator, or other
  carry-over between iterations. False otherwise.
- ``outer_loop_count`` (int): the count of outer-loop iterations (typically
  ``M // TILE_P``). Must be divisible by ``target_grid_size``. Null if not
  applicable.
- ``target_grid_size`` (int): the candidate grid size to predict for. The agent
  fills in 2 for the typical Trainium2 case; sweep multiple candidates by
  re-running the predictor with different values.

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

OPTIMIZATION_NUMBER = 38
OPTIMIZATION_NAME = "SPMD Launch Grid"

REQUIRED_PROFILE_FIELDS = ("t_total_us",)

KNOB_NAMES: dict[str, str] = {
    "has_no_spmd_grid": (
        "True iff the wrapper invokes the kernel as kernel(args) (no launch-grid "
        "syntax) — i.e. nl.num_programs(0) == 1 inside the kernel. Set False "
        "(model returns None) if the kernel is already launched as kernel[G](args) "
        "with G > 1."
    ),
    "has_independent_outer_loop": (
        "True iff the outer sequential_range iterations are independent — no "
        "running scan, accumulator, cross-iteration data dependency, or any "
        "other carry-over between loop iterations. Set False if any iteration "
        "reads state written by a previous iteration."
    ),
    "outer_loop_count": (
        "Count of outer-loop iterations (e.g. M // TILE_P in the typical "
        "tile-row pattern). Must be divisible by target_grid_size for the "
        "shard math to work. Null if not applicable."
    ),
    "target_grid_size": (
        "Candidate SPMD grid size to predict for (the G in kernel[G](...)). "
        "Sweep candidates by setting this to 1, 2 (and optionally 4) and "
        "re-running the predictor; pick the highest-speedup non-extrapolating "
        "prediction. Must be ≥ 1 and ≤ 8 (NKI hard limit)."
    ),
}


@dataclass
class Knobs:
    has_no_spmd_grid: bool
    has_independent_outer_loop: bool
    outer_loop_count: int
    target_grid_size: int


SPMD_GRID_MAX: int = 8

_FIT = load_fit(__file__)


def predict(profile: KernelProfile, knobs: Knobs) -> PredictionDelta | None:
    if not knobs.has_no_spmd_grid:
        return None
    if not knobs.has_independent_outer_loop:
        return None
    if knobs.outer_loop_count is None or knobs.target_grid_size is None:
        return None
    if knobs.target_grid_size < 1 or knobs.target_grid_size > SPMD_GRID_MAX:
        return None
    if knobs.outer_loop_count % knobs.target_grid_size != 0:
        return None
    if profile.t_total_us is None:
        return None

    # v3: filter cases where compute is already saturating — SPMD sharding
    # won't help if t_active >= 95% of t_total (no idle time to recover by
    # parallelising). See benchmark-opt/reports/presentation_slides.md
    # (poorly-calibrated section): opt38 over-predicts when the per-NC time
    # is bounded by saturated compute, not by serial outer iterations.
    if (profile.t_active_us is not None and profile.t_total_us > 0
            and profile.t_active_us / profile.t_total_us > 0.95):
        return None

    G = knobs.target_grid_size
    if G == 1:
        # No-op grid: predict no change.
        return PredictionDelta(
            predicted_after=replace(profile),
            delta_us={"t_total_us": 0.0},
            t_after_us=profile.t_total_us,
            speedup=1.0,
            extrapolation_warning=None,
            notes="target_grid_size=1 — equivalent to no SPMD launch; predicting unchanged latency.",
        )

    alpha_par = _FIT["constants"]["alpha_par"]
    alpha_overhead = _FIT["constants"]["alpha_overhead"]

    r = 1.0 / G
    t_after = alpha_par * profile.t_total_us * r + alpha_overhead * (1.0 - r)

    # v5: DMA-share guard — the two physical NCs share the HBM bus, so the
    # DMA-time fraction of t_total is NOT parallelisable by SPMD. Effective
    # post-shard latency has a floor of:
    #     t_dma + (t_total - t_dma) / G
    # Without this guard, opt38 over-predicts on bandwidth-bound kernels — see
    # benchmark-opt/reports/mg_v2_lncfix_shortfalls.md Case 5 (cumsum, ~92%
    # DMA-bound, 2.0× predicted vs 1.24× measured).
    dma_bound_floor: float | None = None
    dma_note = ""
    if (profile.t_dma_us is not None and profile.t_dma_us > 0
            and profile.t_total_us > 0):
        t_dma = min(profile.t_dma_us, profile.t_total_us)
        t_parallel = max(0.0, profile.t_total_us - t_dma)
        dma_bound_floor = t_dma + t_parallel / G
        dma_frac = t_dma / profile.t_total_us
        if dma_bound_floor > t_after:
            t_after = dma_bound_floor
            dma_note = (
                f" DMA-share guard active: {dma_frac:.0%} of t_total is on the "
                f"shared HBM bus and does not scale with G, so the speedup is "
                f"bounded by t_dma + (t_total - t_dma) / G."
            )
    t_after = max(t_after, 1.0)

    delta_total = t_after - profile.t_total_us
    predicted_after = replace(
        profile,
        t_total_us=t_after,
    )
    n_before = knobs.outer_loop_count
    n_after = knobs.outer_loop_count // G
    return PredictionDelta(
        predicted_after=predicted_after,
        delta_us={"t_total_us": delta_total},
        t_after_us=t_after,
        speedup=profile.t_total_us / t_after if t_after > 0 else float("inf"),
        extrapolation_warning=extrapolation_warning(_FIT, {
            "outer_loop_count": knobs.outer_loop_count,
            "target_grid_size": knobs.target_grid_size,
        }),
        notes=(f"Shards {n_before} outer iterations across G={G} program instances "
               f"({n_after} iters/instance); speedup is bounded by min(G, physical_NCs) "
               f"and by the parallelisable fraction of the kernel." + dma_note),
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

    print(f"{'kernel_id':>30s}  {'n_iters':>7s}  {'G':>3s}  "
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
            has_no_spmd_grid=True,
            has_independent_outer_loop=True,
            outer_loop_count=int(b["outer_loop_count"]),
            target_grid_size=int(a["target_grid_size"]),
        )
        pred = predict(profile, knobs)
        obs_speedup = float(b["t_total_us"]) / float(a["t_total_us"])
        err = abs(pred.speedup - obs_speedup) / obs_speedup * 100
        print(f"{kid:>30s}  {b['outer_loop_count']:>7s}  {a['target_grid_size']:>3s}  "
              f"{pred.speedup:>12.2f}x  {obs_speedup:>11.2f}x  {err:>5.1f}%")
