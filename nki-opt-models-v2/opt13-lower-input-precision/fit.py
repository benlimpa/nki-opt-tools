"""
fit.py — fit alpha coefficients for opt13 from calibration.csv.

Model:
    read_bytes  = (M*K + K*N) * 4
    write_bytes = M*N * 4
    r_dma = (read_bytes * 0.5 + write_bytes) / (read_bytes + write_bytes)
    t_after = alpha_dma   * t_dma_b * r_dma
            + alpha_tensor * t_tensor_b
            + alpha_fixed

Three free parameters fit via 3x3 normal equations against paired rows.
The dtype_ratio for the calibration sweep is fixed at 0.5 (bf16 from f32).
DMA and TensorE active times overlap inside t_total, so a separate t_other
term collapses to ~0 and breaks the fit — α_fixed absorbs the small
scalar/vector residual.
"""

from __future__ import annotations

import csv
import json
import pathlib
import sys

DTYPE_RATIO = 0.5


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
    M_seen, K_seen, N_seen = [], [], []
    for p in paired:
        before, after = p["before"], p["after"]
        M_, K_, N_ = int(before["M"]), int(before["K"]), int(before["N"])
        read_bytes = (M_ * K_ + K_ * N_) * 4
        write_bytes = M_ * N_ * 4
        r_dma = (read_bytes * DTYPE_RATIO + write_bytes) / (read_bytes + write_bytes)
        t_dma = float(before["t_dma_us"])
        t_tensor = float(before["t_tensor_us"])
        rows.append((t_dma * r_dma, t_tensor, 1.0))
        y.append(float(after["t_total_us"]))
        M_seen.append(M_); K_seen.append(K_); N_seen.append(N_)

    if len(paired) >= 3:
        alpha_dma, alpha_tensor, alpha_fixed = _normal_eq(rows, y, 3)
    elif len(paired) == 2:
        # Pin alpha_dma=1.0; solve 2x2 for the rest.
        alpha_dma = 1.0
        a11 = sum(r[1] * r[1] for r in rows)
        a12 = sum(r[1] * r[2] for r in rows)
        a22 = sum(r[2] * r[2] for r in rows)
        b1 = sum(r[1] * (yi - alpha_dma * r[0]) for r, yi in zip(rows, y))
        b2 = sum(r[2] * (yi - alpha_dma * r[0]) for r, yi in zip(rows, y))
        det = a11 * a22 - a12 * a12
        if abs(det) < 1e-12:
            raise ValueError("singular 2-point system")
        alpha_tensor = (a22 * b1 - a12 * b2) / det
        alpha_fixed = (-a12 * b1 + a11 * b2) / det
        print(f"[info] 2-point fit: pinned alpha_dma=1.0")
    else:
        alpha_dma, alpha_tensor, alpha_fixed = 1.0, 0.4, 0.0
        print(f"[info] single-point fit: all coefficients pinned")

    max_err_pct = 0.0
    for (x1, x2, x3), t_after in zip(rows, y):
        pred = alpha_dma * x1 + alpha_tensor * x2 + alpha_fixed * x3
        err_pct = abs(pred - t_after) / t_after * 100
        max_err_pct = max(max_err_pct, err_pct)

    fit = {
        "constants": {
            "alpha_dma": round(alpha_dma, 6),
            "alpha_tensor": round(alpha_tensor, 6),
            "alpha_fixed": round(alpha_fixed, 6),
        },
        "knob_range": {
            "M": [min(M_seen), max(M_seen)],
            "K": [min(K_seen), max(K_seen)],
            "N": [min(N_seen), max(N_seen)],
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
