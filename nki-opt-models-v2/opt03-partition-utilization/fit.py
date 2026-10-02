"""
fit.py — fit alpha coefficients for opt03 from calibration.csv.

Model:
    t_after = alpha_iter * t_total_b * r + alpha_overhead * (1 - r)
where r = TILE_P_current / 128.

Two free parameters fit via 2x2 normal equations against paired rows.
"""

from __future__ import annotations

import csv
import json
import pathlib
import sys

TILE_P_MAX = 128


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


def _lsq_2x2(rows: list[tuple[float, float]], y: list[float]) -> tuple[float, float]:
    a11 = sum(r[0] * r[0] for r in rows)
    a12 = sum(r[0] * r[1] for r in rows)
    a22 = sum(r[1] * r[1] for r in rows)
    b1 = sum(r[0] * yi for r, yi in zip(rows, y))
    b2 = sum(r[1] * yi for r, yi in zip(rows, y))
    det = a11 * a22 - a12 * a12
    if abs(det) < 1e-12:
        raise ValueError("singular system — calibration points degenerate")
    return ((a22 * b1 - a12 * b2) / det,
            (-a12 * b1 + a11 * b2) / det)


def main() -> int:
    here = pathlib.Path(__file__).parent
    csv_path = here / "calibration.csv"
    out_path = here / "fit.json"
    if not csv_path.exists():
        print(f"[error] {csv_path} missing", file=sys.stderr)
        return 1

    paired = _read_paired(csv_path)
    if not paired:
        print(f"[error] no paired rows", file=sys.stderr)
        return 1

    rows: list[tuple[float, float]] = []
    y: list[float] = []
    M_seen, TILE_P_seen = [], []
    for p in paired:
        before, after = p["before"], p["after"]
        TILE_P = int(before["TILE_P_current"])
        r = TILE_P / TILE_P_MAX
        t_b = float(before["t_total_us"])
        rows.append((t_b * r, 1.0 - r))
        y.append(float(after["t_total_us"]))
        M_seen.append(int(before["M"]))
        TILE_P_seen.append(TILE_P)

    if len(paired) < 2:
        # Single-point fallback: pin alpha_iter=1, solve alpha_overhead.
        alpha_iter = 1.0
        x1, x2 = rows[0]
        if abs(x2) < 1e-9:
            raise ValueError("single point with r=1 — degenerate")
        alpha_overhead = (y[0] - alpha_iter * x1) / x2
        print(f"[info] single-point fit: pinned alpha_iter=1.0, "
              f"solved alpha_overhead={alpha_overhead:.4f}")
    else:
        alpha_iter, alpha_overhead = _lsq_2x2(rows, y)

    max_err_pct = 0.0
    for (x1, x2), t_after in zip(rows, y):
        pred = alpha_iter * x1 + alpha_overhead * x2
        err_pct = abs(pred - t_after) / t_after * 100
        max_err_pct = max(max_err_pct, err_pct)

    fit = {
        "constants": {
            "alpha_iter": round(alpha_iter, 6),
            "alpha_overhead": round(alpha_overhead, 6),
        },
        "knob_range": {
            "M": [min(M_seen), max(M_seen)],
            "TILE_P_current": [min(TILE_P_seen), max(TILE_P_seen)],
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
