"""
Performance model for Optimization #5: Loop Fusion — 2-loop / single
intermediate variant.
====================================================================

opt05 is the **strict-subset** specialization of opt07: exactly two
sequential ``affine_range`` loops over the same tile range, separated
by one ``private_hbm`` buffer. After fusion, the intermediate stays
in SBUF and the kernel runs as a single tile loop.

opt05 is mathematically identical to opt07 with ``n_intermediates=1``,
so it reuses opt07's fit. The opt05/opt07 split exists for the
optimizer agent's pattern catalog: opt05's stricter applicability
(exactly 2 loops, 1 intermediate) lets the agent pattern-match the
canonical 2-loop case independently of the more general n-loop fusion.

v2 design — caller-anchored
---------------------------
Same formula as opt07 with ``n_intermediates`` pinned to 1:

    r_dma   = 1 / 2 = 0.5
    t_after = alpha_dma * t_dma_b * r_dma
              + alpha_other * (t_total_b - t_dma_b)
              + alpha_fixed

Constants come from opt07's ``fit.json`` (5 paired observations, 3-param
LSQ; the calibration sweep includes single-intermediate variants).

Knobs
-----
- ``has_hbm_intermediate`` (bool): True iff the kernel has exactly
  one ``private_hbm`` buffer between two sequential affine_range
  loops over the same tile range. Profiler signature:
  ``spill_save_bytes > 0``. Set to false if no HBM intermediate
  exists OR if there are 2+ intermediates (use opt07 instead).
- ``n_tiles`` (int): tile-loop trip count (M // TILE_P). Used for
  applicability + extrapolation.

Required profile fields
-----------------------
``t_total_us``, ``t_dma_us``.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, replace

from _common import (  # type: ignore[import-not-found]
    KernelProfile,
    PredictionDelta,
    extrapolation_warning,
)

OPTIMIZATION_NUMBER = 5
OPTIMIZATION_NAME = "Loop Fusion — 2-loop single-intermediate variant"

REQUIRED_PROFILE_FIELDS = ("t_total_us", "t_dma_us")

KNOB_NAMES: dict[str, str] = {
    "has_hbm_intermediate": (
        "True iff the kernel has exactly one private_hbm buffer used "
        "as an intermediate between two sequential affine_range/"
        "sequential_range loops over the same tile range. Profiler "
        "signature: spill_save_bytes > 0 with spill_reload_bytes > 0. "
        "If there are 2+ intermediates, use opt07 instead."
    ),
    "n_tiles": (
        "Tile-loop trip count (= M // TILE_P). Used for applicability "
        "+ extrapolation — the speedup is governed by the structural "
        "ratio rather than n_tiles. Null if pattern does not apply."
    ),
}


@dataclass
class Knobs:
    has_hbm_intermediate: bool
    n_tiles: int


_FIT_PATH = (pathlib.Path(__file__).parent.parent
             / "opt07-loop-fusion" / "fit.json")
if not _FIT_PATH.exists():
    raise FileNotFoundError(
        f"opt05 reuses opt07's fit; expected {_FIT_PATH} but it is missing."
    )
_FIT = json.loads(_FIT_PATH.read_text())


def infer_applicable_from_profile(
    profile: KernelProfile, opt_features: dict
) -> dict | None:
    """v3 profile-driven applicability override.

    The fusion family's canonical signal is ``spill_save_bytes > 0`` — the
    compiler materialized an intermediate to HBM, which fusion eliminates.
    The agent often fails to set ``has_hbm_intermediate=True`` even when this
    signal is present (see benchmark-opt/reports/mg_v2_predictor_misses.md
    case C2: matmul_add_rmsnorm). If the agent supplied ``n_tiles`` but left
    applicable false, this hook auto-marks the opt applicable from profile.
    """
    if not profile.spill_save_bytes:
        return None
    n_tiles = opt_features.get("n_tiles")
    if not n_tiles or n_tiles < 1:
        return None
    return {"has_hbm_intermediate": True, "n_tiles": n_tiles}


def predict(profile: KernelProfile, knobs: Knobs) -> PredictionDelta | None:
    if not knobs.has_hbm_intermediate:
        return None
    if knobs.n_tiles is None or knobs.n_tiles < 1:
        return None
    if profile.t_dma_us is None:
        return None

    alpha_dma = _FIT["constants"]["alpha_dma"]
    alpha_other = _FIT["constants"]["alpha_other"]
    alpha_fixed = _FIT["constants"]["alpha_fixed"]

    r_dma = 0.5  # exactly one intermediate → fused kernel halves DMA
    t_other_b = max(profile.t_total_us - profile.t_dma_us, 0.0)
    t_after = (alpha_dma * profile.t_dma_us * r_dma
               + alpha_other * t_other_b
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
            "n_intermediates": 1,
            "n_tiles": knobs.n_tiles,
        }),
        notes=("Fuses 2 sequential loops, eliminating 1 HBM intermediate "
               f"({2 * knobs.n_tiles} round-trips removed)"),
    )


if __name__ == "__main__":
    # Self-test: re-evaluate against opt07's calibration rows that have
    # n_intermediates=1 (opt05 is the strict-subset case).
    import csv
    csv_path = (pathlib.Path(__file__).parent.parent
                / "opt07-loop-fusion" / "calibration.csv")
    if not csv_path.exists():
        print("[error] opt07 calibration.csv missing — collect data first")
        raise SystemExit(1)
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    by_id: dict[str, dict[str, dict]] = {}
    for r in rows:
        by_id.setdefault(r["kernel_id"], {})[r["variant"]] = r

    print(f"{'kernel_id':>30s}  {'n_int':>5s}  {'n_tiles':>7s}  "
          f"{'pred speedup':>13s}  {'obs speedup':>12s}  {'err%':>6s}")
    print("-" * 90)
    n_evaluated = 0
    for kid, pair in by_id.items():
        if "before" not in pair or "after" not in pair:
            continue
        b = pair["before"]
        a = pair["after"]
        if int(b["n_intermediates"]) != 1:
            continue  # opt05 only applies to single-intermediate case
        profile = KernelProfile(
            t_total_us=float(b["t_total_us"]),
            t_active_us=float(b.get("t_active_us") or 0),
            t_dma_us=float(b["t_dma_us"]),
        )
        knobs = Knobs(has_hbm_intermediate=True, n_tiles=int(b["n_tiles"]))
        pred = predict(profile, knobs)
        obs_speedup = float(b["t_total_us"]) / float(a["t_total_us"])
        err = abs(pred.speedup - obs_speedup) / obs_speedup * 100
        print(f"{kid:>30s}  {b['n_intermediates']:>5s}  {b['n_tiles']:>7s}  "
              f"{pred.speedup:>12.2f}x  {obs_speedup:>11.2f}x  {err:>5.1f}%")
        n_evaluated += 1

    if n_evaluated == 0:
        print("[warn] no n_intermediates=1 rows in opt07 calibration — "
              "extend the opt07 sweep to include the single-intermediate case")
