#!/usr/bin/env python3
"""
Generate parameterized mean-subtract before/after kernels for the opt21 sweep.

Usage:  python _generate.py

VARIANTS list: (M, N) tuples. TILE_M=128 fixed.

Before kernel: two affine_range loops with private_hbm row-means
intermediate between them.
After kernel: single fused loop; row mean stays in SBUF.
"""

from __future__ import annotations

import pathlib

TILE_M = 128

VARIANTS = [
    # (M, N) — all M divisible by TILE_M=128.
    (512,  1024),  # v1 anchor: n_tiles=4
    (1024, 1024),  # n_tiles=8
    (2048, 1024),  # n_tiles=16
    (512,  512),   # smaller N
    (1024, 2048),  # bigger N
]


_BEFORE_TEMPLATE = '''\
"""
Sweep variant — opt21 BEFORE: two affine_range loops with private_hbm
intermediate (row means).
M={M}, N={N}, TILE_M={TILE_M}.
"""
import nki
import nki.isa as nisa
import nki.language as nl

M = {M}
N = {N}
TILE_M = {TILE_M}


@nki.jit(platform_target="trn1")
def mean_subtract_two_loops(x):
    y = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)
    means_hbm = nl.ndarray((M, 1), dtype=nl.float32, buffer=nl.private_hbm)

    # Loop 1: compute per-row mean, write to HBM
    for m in nl.affine_range(M // TILE_M):
        x_tile = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tile, src=x[m * TILE_M:(m + 1) * TILE_M, 0:N])

        row_sum = nl.ndarray((TILE_M, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=row_sum, data=x_tile, op=nl.add, axis=1, keepdims=True)

        row_mean = nl.ndarray((TILE_M, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=row_mean, data=row_sum,
                           op0=nl.multiply, operand0=1.0 / N)

        nisa.dma_copy(dst=means_hbm[m * TILE_M:(m + 1) * TILE_M, 0:1],
                      src=row_mean)

    # Loop 2: reload x and means, subtract
    for m in nl.affine_range(M // TILE_M):
        x_tile2 = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tile2, src=x[m * TILE_M:(m + 1) * TILE_M, 0:N])

        mean_tile = nl.ndarray((TILE_M, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=mean_tile,
                      src=means_hbm[m * TILE_M:(m + 1) * TILE_M, 0:1])

        y_tile = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=y_tile, data=x_tile2,
                           op0=nl.subtract, operand0=mean_tile)

        nisa.dma_copy(dst=y[m * TILE_M:(m + 1) * TILE_M, 0:N], src=y_tile)

    return y
'''

_AFTER_TEMPLATE = '''\
"""
Sweep variant — opt21 AFTER: single fused affine_range loop, no
private_hbm intermediate.
M={M}, N={N}, TILE_M={TILE_M}.
"""
import nki
import nki.isa as nisa
import nki.language as nl

M = {M}
N = {N}
TILE_M = {TILE_M}


@nki.jit(platform_target="trn1")
def mean_subtract_fused(x):
    y = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    for m in nl.affine_range(M // TILE_M):
        x_tile = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tile, src=x[m * TILE_M:(m + 1) * TILE_M, 0:N])

        row_sum = nl.ndarray((TILE_M, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=row_sum, data=x_tile, op=nl.add, axis=1, keepdims=True)

        row_mean = nl.ndarray((TILE_M, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=row_mean, data=row_sum,
                           op0=nl.multiply, operand0=1.0 / N)

        y_tile = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=y_tile, data=x_tile,
                           op0=nl.subtract, operand0=row_mean)

        nisa.dma_copy(dst=y[m * TILE_M:(m + 1) * TILE_M, 0:N], src=y_tile)

    return y
'''


def main() -> None:
    here = pathlib.Path(__file__).parent
    for M, N in VARIANTS:
        if M % TILE_M != 0:
            raise SystemExit(f"M={M} must be divisible by TILE_M={TILE_M}")
        ctx = dict(M=M, N=N, TILE_M=TILE_M)
        base = f"ms_M{M}_N{N}"
        (here / f"{base}_before.py").write_text(_BEFORE_TEMPLATE.format(**ctx))
        (here / f"{base}_after.py").write_text(_AFTER_TEMPLATE.format(**ctx))
        print(f"  wrote {base}_before.py + {base}_after.py")


if __name__ == "__main__":
    main()
