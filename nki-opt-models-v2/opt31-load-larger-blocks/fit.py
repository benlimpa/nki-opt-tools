"""
fit.py — fit alpha coefficients for opt31 from calibration.csv.

Model:
    t_after = alpha_dma    * t_dma_b    / tile_ratio
            + alpha_vector * t_vector_b / tile_ratio
            + alpha_other  * (t_total_b - t_dma_b - t_vector_b)
            + alpha_fixed

with tile_ratio = N / TILE_N_current. Four free parameters; needs ≥4
non-degenerate paired observations. Solved via least-squares (4x4 normal
equations).

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


def _gauss_solve(M: list[list[float]], rhs: list[float]) -> list[float]:
    """Gaussian elimination on a square system. Tiny + dependency-free."""
    n = len(rhs)
    A = [row[:] + [r] for row, r in zip(M, rhs)]
    for i in range(n):
        pivot = max(range(i, n), key=lambda r: abs(A[r][i]))
        A[i], A[pivot] = A[pivot], A[i]
        if abs(A[i][i]) < 1e-12:
            raise ValueError("singular system — calibration points are degenerate")
        for j in range(i + 1, n):
            factor = A[j][i] / A[i][i]
            for k in range(i, n + 1):
                A[j][k] -= factor * A[i][k]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (A[i][n] - sum(A[i][j] * x[j] for j in range(i + 1, n))) / A[i][i]
    return x


def _lsq(rows: list[tuple[float, ...]], y: list[float], k: int) -> list[float]:
    """Fit y = sum(coef[i] * row[i]) via k×k normal equations."""
    n = len(rows)
    AtA = [[sum(r[i] * r[j] for r in rows) for j in range(k)] for i in range(k)]
    Atb = [sum(r[i] * yi for r, yi in zip(rows, y)) for i in range(k)]
    return _gauss_solve(AtA, Atb)


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

    rows: list[tuple[float, float, float, float]] = []  # (dma/r, vec/r, other, 1)
    y: list[float] = []
    N_seen, ratio_seen = [], []
    for p in paired:
        before = p["before"]
        after = p["after"]
        t_total_b = float(before["t_total_us"])
        t_dma_b = float(before["t_dma_us"])
        t_vector_b = float(before["t_vector_us"])
        t_other = max(t_total_b - t_dma_b - t_vector_b, 0.0)
        N = int(before["N"])
        tile_n = int(before["TILE_N_current"])
        ratio = N / tile_n
        rows.append((t_dma_b / ratio, t_vector_b / ratio, t_other, 1.0))
        y.append(float(after["t_total_us"]))
        N_seen.append(N)
        ratio_seen.append(ratio)

    if len(paired) < 4:
        # Underdetermined — pin alpha_dma=1, alpha_vector=1, alpha_other=1
        # and solve for alpha_fixed only.
        a_dma, a_vec, a_other = 1.0, 1.0, 1.0
        residuals = [t_after - (a_dma * x1 + a_vec * x2 + a_other * x3)
                     for (x1, x2, x3, _), t_after in zip(rows, y)]
        a_fixed = sum(residuals) / len(residuals)
        coefs = [a_dma, a_vec, a_other, a_fixed]
        print(f"[info] underdetermined fit ({len(paired)} pairs): pinned "
              f"alpha_dma=alpha_vector=alpha_other=1, solved alpha_fixed={a_fixed:.4f}")
    else:
        coefs = _lsq(rows, y, 4)

    a_dma, a_vec, a_other, a_fixed = coefs

    max_err_pct = 0.0
    for (x1, x2, x3, _), t_after in zip(rows, y):
        pred = a_dma * x1 + a_vec * x2 + a_other * x3 + a_fixed
        err_pct = abs(pred - t_after) / t_after * 100
        max_err_pct = max(max_err_pct, err_pct)

    fit = {
        "constants": {
            "alpha_dma": round(a_dma, 6),
            "alpha_vector": round(a_vec, 6),
            "alpha_other": round(a_other, 6),
            "alpha_fixed": round(a_fixed, 6),
        },
        "knob_range": {
            "N": [min(N_seen), max(N_seen)],
            "tile_ratio": [min(ratio_seen), max(ratio_seen)],
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
