"""
fit.py — fit alpha coefficients for opt34 from calibration.csv.

Model:
    t_after = beta_dma * t_dma_b
            + beta_par * (t_total_b - t_dma_b) / P
            + alpha_fixed

with P = 128 (hardware partition count). Three free parameters fit via
least-squares (3x3 normal equations) against paired (before, after) rows.

Output: fit.json with constants, knob_range, max % error, n_points.
"""

from __future__ import annotations

import csv
import json
import pathlib
import sys

P_HARDWARE = 128


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


def _solve_3x3(M: list[list[float]], rhs: list[float]) -> list[float]:
    A = [row[:] + [r] for row, r in zip(M, rhs)]
    n = 3
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


def _lsq_3param(rows: list[tuple[float, float, float]], y: list[float]) -> tuple[float, float, float]:
    """Fit y = a*x1 + b*x2 + c via 3x3 normal equations."""
    n = len(rows)
    s11 = sum(r[0] * r[0] for r in rows)
    s22 = sum(r[1] * r[1] for r in rows)
    s33 = float(n)
    s12 = sum(r[0] * r[1] for r in rows)
    s13 = sum(r[0] for r in rows)
    s23 = sum(r[1] for r in rows)
    b1 = sum(r[0] * yi for r, yi in zip(rows, y))
    b2 = sum(r[1] * yi for r, yi in zip(rows, y))
    b3 = sum(y)
    M = [[s11, s12, s13], [s12, s22, s23], [s13, s23, s33]]
    sol = _solve_3x3(M, [b1, b2, b3])
    return sol[0], sol[1], sol[2]


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

    rows: list[tuple[float, float, float]] = []  # (t_dma_b, t_serial/P, 1)
    y: list[float] = []
    N_seen = []
    for p in paired:
        before = p["before"]
        after = p["after"]
        t_total_b = float(before["t_total_us"])
        t_dma_b = float(before["t_dma_us"])
        t_serial = max(t_total_b - t_dma_b, 0.0)
        t_after = float(after["t_total_us"])
        rows.append((t_dma_b, t_serial / P_HARDWARE, 1.0))
        y.append(t_after)
        N_seen.append(int(before["N_total"]))

    if len(paired) < 3:
        # Underdetermined — pin beta_dma=0.5 (bulk DMA halves packet overhead),
        # beta_par=1.0 (perfect parallelization). Solve for alpha_fixed.
        beta_dma = 0.5
        beta_par = 1.0
        residuals = [t_after - (beta_dma * x1 + beta_par * x2)
                     for (x1, x2, _), t_after in zip(rows, y)]
        alpha_fixed = sum(residuals) / len(residuals)
        print(f"[info] underdetermined fit ({len(paired)} pairs): pinned "
              f"beta_dma=0.5, beta_par=1.0, solved alpha_fixed={alpha_fixed:.4f}")
    else:
        beta_dma, beta_par, alpha_fixed = _lsq_3param(rows, y)

    max_err_pct = 0.0
    for (x1, x2, _), t_after in zip(rows, y):
        pred = beta_dma * x1 + beta_par * x2 + alpha_fixed
        err_pct = abs(pred - t_after) / t_after * 100
        max_err_pct = max(max_err_pct, err_pct)

    fit = {
        "constants": {
            "beta_dma": round(beta_dma, 6),
            "beta_par": round(beta_par, 6),
            "alpha_fixed": round(alpha_fixed, 6),
        },
        "knob_range": {
            "N_total": [min(N_seen), max(N_seen)],
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
