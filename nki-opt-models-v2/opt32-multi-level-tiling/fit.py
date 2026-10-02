"""
fit.py — fit alpha coefficients for opt32 from calibration.csv.

Model:
    a_bytes = M*K*4
    b_bytes = K*N*4
    write_bytes = M*N*4
    read_before = a_bytes * n_n_tiles + b_bytes
    read_after  = a_bytes + b_bytes
    r_dma_after = (read_after + write_bytes) / (read_before + write_bytes)

    t_after = alpha_dma * t_dma_b * r_dma_after
            + alpha_other * t_total_b
            + alpha_fixed

Three free parameters fit via 3x3 normal equations against paired rows.
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


def _solve_3x3(A, b):
    n = 3
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


def _normal_eq(rows, y):
    A = [[sum(r[i] * r[j] for r in rows) for j in range(3)] for i in range(3)]
    b = [sum(r[i] * yi for r, yi in zip(rows, y)) for i in range(3)]
    x = _solve_3x3(A, b)
    return x[0], x[1], x[2]


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
    M_seen, K_seen, N_seen, n_n_seen = [], [], [], []
    for p in paired:
        before, after = p["before"], p["after"]
        M_, K_, N_ = int(before["M"]), int(before["K"]), int(before["N"])
        n_n = int(before["n_n_tiles"])
        a_bytes = M_ * K_ * 4
        b_bytes = K_ * N_ * 4
        write_bytes = M_ * N_ * 4
        read_before = a_bytes * n_n + b_bytes
        read_after = a_bytes + b_bytes
        r_dma_after = (read_after + write_bytes) / (read_before + write_bytes)
        t_dma = float(before["t_dma_us"])
        t_total = float(before["t_total_us"])
        rows.append((t_dma * r_dma_after, t_total, 1.0))
        y.append(float(after["t_total_us"]))
        M_seen.append(M_); K_seen.append(K_); N_seen.append(N_); n_n_seen.append(n_n)

    if len(paired) >= 3:
        alpha_dma, alpha_other, alpha_fixed = _normal_eq(rows, y)
    elif len(paired) == 2:
        alpha_dma = 1.0
        a11 = sum(r[1] * r[1] for r in rows)
        a12 = sum(r[1] * r[2] for r in rows)
        a22 = sum(r[2] * r[2] for r in rows)
        b1 = sum(r[1] * (yi - alpha_dma * r[0]) for r, yi in zip(rows, y))
        b2 = sum(r[2] * (yi - alpha_dma * r[0]) for r, yi in zip(rows, y))
        det = a11 * a22 - a12 * a12
        if abs(det) < 1e-12:
            raise ValueError("singular 2-point system")
        alpha_other = (a22 * b1 - a12 * b2) / det
        alpha_fixed = (-a12 * b1 + a11 * b2) / det
        print(f"[info] 2-point fit: pinned alpha_dma=1.0")
    else:
        alpha_dma, alpha_other, alpha_fixed = 1.0, 0.0, 0.0
        print(f"[info] single-point fit: pinned alpha_dma=1.0, others=0")

    max_err_pct = 0.0
    for (x1, x2, x3), t_after in zip(rows, y):
        pred = alpha_dma * x1 + alpha_other * x2 + alpha_fixed * x3
        err_pct = abs(pred - t_after) / t_after * 100
        max_err_pct = max(max_err_pct, err_pct)

    fit = {
        "constants": {
            "alpha_dma": round(alpha_dma, 6),
            "alpha_other": round(alpha_other, 6),
            "alpha_fixed": round(alpha_fixed, 6),
        },
        "knob_range": {
            "M": [min(M_seen), max(M_seen)],
            "K": [min(K_seen), max(K_seen)],
            "N": [min(N_seen), max(N_seen)],
            "n_n_tiles": [min(n_n_seen), max(n_n_seen)],
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
