"""
fit.py — fit alpha coefficients for opt16 from calibration.csv.

Model:
    t_after = alpha_dma * t_dma + alpha_overhead * t_overhead
    t_overhead = t_total_before - t_dma_before - t_tensor_before

We fit the alphas via least-squares against paired (before, after) rows in
calibration.csv. With ≥2 pairs the system is overdetermined.

Output: fit.json with constants, knob_range, max % error, n_points.
"""

from __future__ import annotations

import csv
import json
import pathlib
import sys


def _read_paired(csv_path: pathlib.Path) -> list[dict]:
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    by_id: dict[str, dict[str, dict]] = {}
    for r in rows:
        by_id.setdefault(r["kernel_id"], {})[r["variant"]] = r
    paired = []
    for kid, pair in by_id.items():
        if "before" in pair and "after" in pair:
            paired.append({"kernel_id": kid, "before": pair["before"], "after": pair["after"]})
    return paired


def _lsq_2x2(A_rows: list[tuple[float, float]], b: list[float]) -> tuple[float, float]:
    """
    Least-squares solve for [α_dma, α_overhead]:  A · [α_dma, α_overhead]ᵀ = b.

    Avoids importing numpy: solve the 2×2 normal equations Aᵀ A x = Aᵀ b directly.
    """
    a11 = sum(a*a for a, _ in A_rows)
    a12 = sum(a*c for a, c in A_rows)
    a22 = sum(c*c for _, c in A_rows)
    bx1 = sum(a*y for (a, _), y in zip(A_rows, b))
    bx2 = sum(c*y for (_, c), y in zip(A_rows, b))
    det = a11 * a22 - a12 * a12
    if abs(det) < 1e-12:
        raise ValueError("singular — need at least 2 non-degenerate pairs")
    inv11 = a22 / det
    inv12 = -a12 / det
    inv22 = a11 / det
    return (inv11 * bx1 + inv12 * bx2,
            inv12 * bx1 + inv22 * bx2)


def main() -> int:
    here = pathlib.Path(__file__).parent
    csv_path = here / "calibration.csv"
    out_path = here / "fit.json"
    if not csv_path.exists():
        print(f"[error] {csv_path} missing", file=sys.stderr)
        return 1

    paired = _read_paired(csv_path)
    if not paired:
        print(f"[error] no paired rows found in {csv_path}", file=sys.stderr)
        return 1

    A_rows: list[tuple[float, float]] = []
    b: list[float] = []
    n_tiles_seen = []
    for p in paired:
        before = p["before"]
        after = p["after"]
        t_total_b = float(before["t_total_us"])
        t_dma_b = float(before["t_dma_us"])
        t_tensor_b = float(before["t_tensor_us"])
        t_overhead = t_total_b - t_dma_b - t_tensor_b
        t_after = float(after["t_total_us"])
        A_rows.append((t_dma_b, t_overhead))
        b.append(t_after)
        n_tiles_seen.append(int(before["n_tiles"]))

    if len(paired) == 1:
        # Underdetermined: pin alpha_dma to 1.0 (DMA active time is preserved
        # by the pipeline) and solve for alpha_overhead from the single point.
        t_dma_b, t_overhead = A_rows[0]
        t_after = b[0]
        if t_overhead <= 0:
            print(f"[error] single-point fit needs t_overhead > 0", file=sys.stderr)
            return 1
        alpha_dma = 1.0
        alpha_overhead = (t_after - alpha_dma * t_dma_b) / t_overhead
        print(f"[info] single-point fit: pinned alpha_dma=1.0, "
              f"solved alpha_overhead={alpha_overhead:.4f}")
    else:
        alpha_dma, alpha_overhead = _lsq_2x2(A_rows, b)

    # Compute residuals: predicted t_after vs observed
    max_err_pct = 0.0
    for (t_dma_b, t_overhead), t_after in zip(A_rows, b):
        pred = alpha_dma * t_dma_b + alpha_overhead * t_overhead
        err_pct = abs(pred - t_after) / t_after * 100
        max_err_pct = max(max_err_pct, err_pct)

    fit = {
        "constants": {
            "alpha_dma": round(alpha_dma, 6),
            "alpha_overhead": round(alpha_overhead, 6),
        },
        "knob_range": {
            "n_tiles": [min(n_tiles_seen), max(n_tiles_seen)],
        },
        "max_pct_err": round(max_err_pct, 3),
        "n_points": len(paired),
    }
    out_path.write_text(json.dumps(fit, indent=2) + "\n")
    print(json.dumps(fit, indent=2))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
