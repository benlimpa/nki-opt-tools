"""
Performance model for Optimization #10: Delay Softmax Division
==============================================================

Pattern: a multi-pass row-wise softmax stores ``exp(x - max)`` to
``private_hbm`` during the reduction pass and reloads it in the
normalization pass. Delaying the division eliminates that
intermediate — pass 3 recomputes ``exp(x - max)`` from x in SBUF
instead of reloading exp from HBM. The savings: one tile-write +
one tile-read of exp per column tile.

v2 design — caller-anchored
---------------------------
Per-tile savings are fixed (one write + one read of P×TILE_N bytes).
Total savings scale linearly with ``n_tiles``. The rest of the
kernel cost is preserved:

    t_after = alpha_total * t_total_b + alpha_iter * n_tiles + alpha_fixed

Three free parameters; fit by 3x3 LSQ. ``alpha_iter`` is expected to
be negative (each tile's intermediate elimination saves time).

Knobs
-----
- ``has_premature_exp_store`` (bool): True iff the kernel writes
  ``exp(x - max)`` to ``private_hbm`` during reduction.
- ``n_tiles`` (int): number of column tiles in the softmax loop
  (= ``N_total // TILE_N``).

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

OPTIMIZATION_NUMBER = 10
OPTIMIZATION_NAME = "Delay Softmax Division"

REQUIRED_PROFILE_FIELDS = ("t_total_us",)

KNOB_NAMES: dict[str, str] = {
    "has_premature_exp_store": (
        "True iff the kernel stores exp(x - max) to nl.private_hbm during "
        "the softmax reduction pass and reloads it for the normalization "
        "pass (profiler shows spill_save_bytes > 0 during the reduction). "
        "Set to false if division is already delayed in pass 3."
    ),
    "n_tiles": (
        "Number of column tiles in the softmax loop "
        "(= N_total // TILE_N). Each tile contributes one write + one "
        "read of an exp intermediate that gets eliminated by the "
        "optimization. Null if the pattern does not apply."
    ),
}


@dataclass
class Knobs:
    has_premature_exp_store: bool
    n_tiles: int


_FIT = load_fit(__file__)


def predict(profile: KernelProfile, knobs: Knobs) -> PredictionDelta | None:
    if not knobs.has_premature_exp_store:
        return None
    if knobs.n_tiles is None or knobs.n_tiles <= 0:
        return None
    if profile.t_total_us is None:
        return None

    alpha_total = _FIT["constants"]["alpha_total"]
    alpha_iter = _FIT["constants"]["alpha_iter"]
    alpha_fixed = _FIT["constants"]["alpha_fixed"]

    t_after = (alpha_total * profile.t_total_us
               + alpha_iter * knobs.n_tiles
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
            "n_tiles": knobs.n_tiles,
        }),
        notes=(f"Eliminates {knobs.n_tiles} exp-intermediate "
               f"write+reload round-trips by delaying division to pass 3"),
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

    print(f"{'kernel_id':>30s}  {'n_tiles':>7s}  "
          f"{'pred speedup':>13s}  {'obs speedup':>12s}  {'err%':>6s}")
    print("-" * 80)
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
            has_premature_exp_store=True,
            n_tiles=int(b["n_tiles"]),
        )
        pred = predict(profile, knobs)
        obs_speedup = float(b["t_total_us"]) / float(a["t_total_us"])
        err = abs(pred.speedup - obs_speedup) / obs_speedup * 100
        print(f"{kid:>30s}  {b['n_tiles']:>7s}  "
              f"{pred.speedup:>12.2f}x  {obs_speedup:>11.2f}x  {err:>5.1f}%")
