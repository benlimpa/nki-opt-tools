"""
fit.py — fit alpha coefficients for opt18 from calibration.csv.

Model:
    t_after = alpha_tensor * t_tensor_b / tile_k_ratio
            + alpha_dma    * t_dma_b    / tile_k_ratio
            + alpha_other  * (t_total_b - t_tensor_b - t_dma_b)
            + alpha_fixed

Four free parameters fit via 4x4 normal equations against paired rows.
"""

from __future__ import annotations

import csv
import json
import pathlib
import sys

TILE_K_TARGET = 128


def _read_paired(csv_path: pathlib.Path) -> list[dict]:
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    by_id: dict[str, dict[str, dict]] = {}
    for r in rows:
        by_id.setdefault(r["kernel_id"], {})[r["variant"]] = r
    return [{"kernel_id": kid, "before": pair["before"], "after": pair["after"]}
            for kid, pair in by_id.items() if "before" in pair and "after" in pair]


def _gauss_solve(M: list[list[float]], rhs: list[float]) -> list[float]:
    n = len(rhs)
    A = [row[:] + [r] for row, r in zip(M, rhs)]
    for i in range(n):
        pivot = max(range(i, n), key=lambda r: abs(A[r][i]))
        A[i], A[pivot] = A[pivot], A[i]
        if abs(A[i][i]) < 1e-12:
            raise ValueError("singular system")
        for j in range(i + 1, n):
            factor = A[j][i] / A[i][i]
            for k in range(i, n + 1):
                A[j][k] -= factor * A[i][k]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (A[i][n] - sum(A[i][j] * x[j] for j in range(i + 1, n))) / A[i][i]
    return x


def _lsq(rows: list[tuple[float, ...]], y: list[float], k: int) -> list[float]:
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
        print(f"[error] no paired rows", file=sys.stderr)
        return 1

    rows: list[tuple[float, float, float, float]] = []
    y: list[float] = []
    K_seen, ratio_seen = [], []
    for p in paired:
        before, after = p["before"], p["after"]
        t_total_b = float(before["t_total_us"])
        t_tensor_b = float(before["t_tensor_us"])
        t_dma_b = float(before["t_dma_us"])
        t_other = max(t_total_b - t_tensor_b - t_dma_b, 0.0)
        K = int(before["K"])
        tile_k = int(before["TILE_K_current"])
        ratio = TILE_K_TARGET / tile_k
        rows.append((t_tensor_b / ratio, t_dma_b / ratio, t_other, 1.0))
        y.append(float(after["t_total_us"]))
        K_seen.append(K)
        ratio_seen.append(ratio)

    if len(paired) < 4:
        a_t, a_d, a_o = 1.0, 1.0, 1.0
        residuals = [t - (a_t * x1 + a_d * x2 + a_o * x3)
                     for (x1, x2, x3, _), t in zip(rows, y)]
        a_f = sum(residuals) / len(residuals)
        coefs = [a_t, a_d, a_o, a_f]
        print(f"[info] underdetermined ({len(paired)}<4): pinned alphas=1, "
              f"alpha_fixed={a_f:.4f}")
    else:
        coefs = _lsq(rows, y, 4)

    a_t, a_d, a_o, a_f = coefs
    max_err_pct = 0.0
    for (x1, x2, x3, _), t in zip(rows, y):
        pred = a_t * x1 + a_d * x2 + a_o * x3 + a_f
        max_err_pct = max(max_err_pct, abs(pred - t) / t * 100)

    fit = {
        "constants": {
            "alpha_tensor": round(a_t, 6),
            "alpha_dma": round(a_d, 6),
            "alpha_other": round(a_o, 6),
            "alpha_fixed": round(a_f, 6),
        },
        "knob_range": {
            "K": [min(K_seen), max(K_seen)],
            "tile_k_ratio": [min(ratio_seen), max(ratio_seen)],
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
