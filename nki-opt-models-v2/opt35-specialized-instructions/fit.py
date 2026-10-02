"""
fit.py — fit alpha coefficients for opt35 from calibration.csv.

Model:
    t_after = alpha_dma * t_dma_b + alpha_iter * num_outer_iters + alpha_fixed

Three free parameters fit via least-squares (3x3 normal equations) against
paired (before, after) rows. Requires ≥3 non-degenerate points.

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


def _solve_3x3(M: list[list[float]], rhs: list[float]) -> list[float]:
    """Gaussian elimination on a 3x3 system. Tiny + dependency-free."""
    A = [row[:] + [r] for row, r in zip(M, rhs)]
    n = 3
    for i in range(n):
        # Pivot
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
    """
    Least-squares fit y = a*x1 + b*x2 + c via 3x3 normal equations
    A^T A [a,b,c]^T = A^T y, where each row of A is [x1, x2, 1].
    """
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

    rows: list[tuple[float, float, float]] = []  # (t_dma_b, num_outer_iters, 1)
    y: list[float] = []
    N_seen, ops_seen, iters_seen = [], [], []
    for p in paired:
        before = p["before"]
        after = p["after"]
        t_dma_b = float(before["t_dma_us"])
        iters = int(before["num_outer_iters"])
        t_after = float(after["t_total_us"])
        rows.append((t_dma_b, float(iters), 1.0))
        y.append(t_after)
        N_seen.append(int(before["broadcast_dim_N"]))
        ops_seen.append(int(before["fused_ops"]))
        iters_seen.append(iters)

    if len(paired) < 3:
        # Underdetermined — pin alpha_dma=1 and alpha_iter=0.02 (v1 estimate),
        # solve for alpha_fixed only.
        alpha_dma = 1.0
        alpha_iter = 0.02
        residuals = [t_after - (alpha_dma * t_dma + alpha_iter * iters)
                     for (t_dma, iters, _), t_after in zip(rows, y)]
        alpha_fixed = sum(residuals) / len(residuals)
        print(f"[info] underdetermined fit ({len(paired)} pairs): pinned "
              f"alpha_dma=1.0, alpha_iter=0.02, solved alpha_fixed={alpha_fixed:.4f}")
    else:
        alpha_dma, alpha_iter, alpha_fixed = _lsq_3param(rows, y)

    # Compute residuals: predicted t_after vs observed
    max_err_pct = 0.0
    for (t_dma, iters, _), t_after in zip(rows, y):
        pred = alpha_dma * t_dma + alpha_iter * iters + alpha_fixed
        err_pct = abs(pred - t_after) / t_after * 100
        max_err_pct = max(max_err_pct, err_pct)

    fit = {
        "constants": {
            "alpha_dma": round(alpha_dma, 6),
            "alpha_iter": round(alpha_iter, 6),
            "alpha_fixed": round(alpha_fixed, 6),
        },
        "knob_range": {
            "broadcast_dim_N": [min(N_seen), max(N_seen)],
            "fused_ops": [min(ops_seen), max(ops_seen)],
            "num_outer_iters": [min(iters_seen), max(iters_seen)],
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
