"""
fit.py — fit alpha coefficients for opt24 from calibration.csv.

Model:
    t_other = t_total_b - t_dma_b - t_vector_b
    t_after = alpha_dma * t_dma_b
            + alpha_vector * t_vector_b
            + alpha_other * t_other
            + alpha_fixed

Four free parameters fit via 4x4 normal equations against paired rows.
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


def _solve_nxn(A, b):
    n = len(A)
    M = [row[:] + [bi] for row, bi in zip(A, b)]
    for k in range(n):
        pivot = max(range(k, n), key=lambda i: abs(M[i][k]))
        if abs(M[pivot][k]) < 1e-12:
            raise ValueError("singular system — calibration points degenerate")
        M[k], M[pivot] = M[pivot], M[k]
        for i in range(k + 1, n):
            factor = M[i][k] / M[k][k]
            for j in range(k, n + 1):
                M[i][j] -= factor * M[k][j]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (M[i][n] - sum(M[i][j] * x[j] for j in range(i + 1, n))) / M[i][i]
    return x


def _normal_eq(rows, y, n_params):
    A = [[sum(r[i] * r[j] for r in rows) for j in range(n_params)] for i in range(n_params)]
    b = [sum(r[i] * yi for r, yi in zip(rows, y)) for i in range(n_params)]
    return _solve_nxn(A, b)


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

    rows = []
    y = []
    P_seen, N_seen, n_int_seen = [], [], []
    for p in paired:
        before, after = p["before"], p["after"]
        t_dma = float(before["t_dma_us"])
        t_vec = float(before["t_vector_us"])
        t_total = float(before["t_total_us"])
        t_other = max(t_total - t_dma - t_vec, 0.0)
        rows.append((t_dma, t_vec, t_other, 1.0))
        y.append(float(after["t_total_us"]))
        P_seen.append(int(before["P"]))
        N_seen.append(int(before["N"]))
        n_int_seen.append(int(before["n_fp32_intermediates"]))

    if len(paired) >= 4:
        alpha_dma, alpha_vector, alpha_other, alpha_fixed = _normal_eq(rows, y, 4)
    elif len(paired) >= 2:
        # Pin alpha_other=1.0; solve 3x3 for the rest.
        alpha_other = 1.0
        rows3 = [(r[0], r[1], r[3]) for r in rows]
        y3 = [yi - alpha_other * r[2] for r, yi in zip(rows, y)]
        if len(paired) >= 3:
            alpha_dma, alpha_vector, alpha_fixed = _normal_eq(rows3, y3, 3)
        else:
            # 2-point: pin alpha_vector=0.58 too.
            alpha_vector = 0.58
            r = rows[0]
            # 1-eq, 1-unknown alpha_dma + alpha_fixed: average over 2.
            alpha_dma = 0.7
            alpha_fixed = sum(yi - alpha_dma * r[0] - alpha_vector * r[1]
                              for r, yi in zip(rows, y)) / len(rows)
        print(f"[info] underdetermined fit: pinned alpha_other=1.0")
    else:
        alpha_dma, alpha_vector, alpha_other, alpha_fixed = 0.7, 0.58, 1.0, 0.0
        print(f"[info] single-point fit: all coefficients pinned")

    max_err_pct = 0.0
    for (x1, x2, x3, x4), t_after in zip(rows, y):
        pred = (alpha_dma * x1 + alpha_vector * x2
                + alpha_other * x3 + alpha_fixed * x4)
        err_pct = abs(pred - t_after) / t_after * 100
        max_err_pct = max(max_err_pct, err_pct)

    fit = {
        "constants": {
            "alpha_dma": round(alpha_dma, 6),
            "alpha_vector": round(alpha_vector, 6),
            "alpha_other": round(alpha_other, 6),
            "alpha_fixed": round(alpha_fixed, 6),
        },
        "knob_range": {
            "P": [min(P_seen), max(P_seen)],
            "N": [min(N_seen), max(N_seen)],
            "n_fp32_intermediates": [min(n_int_seen), max(n_int_seen)],
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
