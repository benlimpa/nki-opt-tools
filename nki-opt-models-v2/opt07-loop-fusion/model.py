"""
Performance model for Optimization #7: Loop Fusion — Eliminate HBM
Intermediates
==================================================================

Pattern: a kernel runs an N-step element-wise pipeline (e.g.
``y = (x * 2) + 1`` then ``* 0.5``) as **N separate** ``affine_range`` loops
with ``private_hbm`` buffers between them. Each inter-loop boundary forces
a full HBM store + reload of the working tensor. Fusing the loops keeps
intermediates in SBUF — total HBM traffic drops from ``(n_intermediates+1)``
read+write passes to a single read+write pass.

v2 design — caller-anchored
---------------------------
DMA cost shrinks by a structural ratio governed by ``n_intermediates``;
non-DMA cost (scalar/vector compute) is approximately preserved:

    r_dma   = 1 / (n_intermediates + 1)
    t_after = alpha_dma * t_dma_b * r_dma
              + alpha_other * (t_total_b - t_dma_b)
              + alpha_fixed

Three free parameters; fit by 3x3 LSQ. The structural DMA ratio is
baked in, so ``alpha_dma`` should fit close to 1.0 if the model is
well-specified.

Knobs
-----
- ``has_hbm_intermediate`` (bool): True iff there are ≥1
  ``private_hbm`` buffers between sequential loops over the same tile
  range. Profiler signature: ``spill_save_bytes > 0`` and
  ``spill_reload_bytes > 0``. Set to false if all loops are already
  fused.
- ``n_intermediates`` (int): number of HBM intermediates to be
  eliminated (= number of inter-loop boundaries to fuse out). For 3
  separate loops, this is 2.
- ``n_tiles`` (int): tile-loop trip count (M // TILE_P). Used only
  for applicability + extrapolation.

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

OPTIMIZATION_NUMBER = 7
OPTIMIZATION_NAME = "Loop Fusion — Eliminate HBM Intermediates"

REQUIRED_PROFILE_FIELDS = ("t_total_us", "t_dma_us")

KNOB_NAMES: dict[str, str] = {
    "has_hbm_intermediate": (
        "True iff the kernel has ≥1 private_hbm buffer used as an "
        "intermediate between two sequential affine_range/sequential_range "
        "loops over the same tile range. Profiler signature: "
        "spill_save_bytes > 0 with spill_reload_bytes > 0. Set to false "
        "if all loops are already fused."
    ),
    "n_intermediates": (
        "Number of HBM intermediates that will be eliminated by fusion. "
        "For K sequential loops sharing the same tile range, this is K-1 "
        "(2 for the canonical 3-loop pipeline). Null if pattern does "
        "not apply."
    ),
    "n_tiles": (
        "Tile-loop trip count (= M // TILE_P). Used only for "
        "applicability + extrapolation — the speedup is governed by "
        "n_intermediates rather than n_tiles. Null if pattern does "
        "not apply."
    ),
}


@dataclass
class Knobs:
    has_hbm_intermediate: bool
    n_intermediates: int
    n_tiles: int


_FIT = load_fit(__file__)


def infer_applicable_from_profile(
    profile: KernelProfile, opt_features: dict
) -> dict | None:
    """v3 profile-driven applicability override.

    Same signal as opt05 (``spill_save_bytes > 0``). opt07 requires
    ``n_intermediates`` from the agent (we cannot infer how many private_hbm
    buffers exist from profile alone), so we only auto-apply when the agent
    supplied both ``n_intermediates >= 1`` and ``n_tiles >= 1``.
    """
    if not profile.spill_save_bytes:
        return None
    n_intermediates = opt_features.get("n_intermediates")
    n_tiles = opt_features.get("n_tiles")
    if not n_intermediates or n_intermediates < 1:
        return None
    if not n_tiles or n_tiles < 1:
        return None
    return {
        "has_hbm_intermediate": True,
        "n_intermediates": n_intermediates,
        "n_tiles": n_tiles,
    }


def predict(profile: KernelProfile, knobs: Knobs) -> PredictionDelta | None:
    if not knobs.has_hbm_intermediate:
        return None
    if knobs.n_intermediates is None or knobs.n_intermediates < 1:
        return None
    if knobs.n_tiles is None or knobs.n_tiles < 1:
        return None
    if profile.t_dma_us is None:
        return None

    alpha_dma = _FIT["constants"]["alpha_dma"]
    alpha_other = _FIT["constants"]["alpha_other"]
    alpha_fixed = _FIT["constants"]["alpha_fixed"]

    r_dma = 1.0 / (knobs.n_intermediates + 1)
    t_other_b = max(profile.t_total_us - profile.t_dma_us, 0.0)
    t_after = (alpha_dma * profile.t_dma_us * r_dma
               + alpha_other * t_other_b
               + alpha_fixed)
    t_after = max(t_after, 1.0)

    delta_total = t_after - profile.t_total_us
    predicted_after = replace(
        profile,
        t_total_us=t_after,
    )
    return PredictionDelta(
        predicted_after=predicted_after,
        delta_us={"t_total_us": delta_total},
        t_after_us=t_after,
        speedup=profile.t_total_us / t_after if t_after > 0 else float("inf"),
        extrapolation_warning=extrapolation_warning(_FIT, {
            "n_intermediates": knobs.n_intermediates,
            "n_tiles": knobs.n_tiles,
        }),
        notes=(f"Fuses {knobs.n_intermediates + 1} sequential loops, "
               f"eliminating {knobs.n_intermediates} HBM intermediates "
               f"({2 * knobs.n_intermediates * knobs.n_tiles} round-trips removed)"),
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

    print(f"{'kernel_id':>30s}  {'n_int':>5s}  {'n_tiles':>7s}  "
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
        )
        knobs = Knobs(
            has_hbm_intermediate=True,
            n_intermediates=int(b["n_intermediates"]),
            n_tiles=int(b["n_tiles"]),
        )
        pred = predict(profile, knobs)
        obs_speedup = float(b["t_total_us"]) / float(a["t_total_us"])
        err = abs(pred.speedup - obs_speedup) / obs_speedup * 100
        print(f"{kid:>30s}  {b['n_intermediates']:>5s}  {b['n_tiles']:>7s}  "
              f"{pred.speedup:>12.2f}x  {obs_speedup:>11.2f}x  {err:>5.1f}%")
