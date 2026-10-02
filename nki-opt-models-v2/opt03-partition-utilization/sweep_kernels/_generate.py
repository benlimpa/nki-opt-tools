#!/usr/bin/env python3
"""
Generate parameterized element-wise add before/after kernels for the opt03
sweep.

Usage:  python _generate.py

VARIANTS list: (M, N, TILE_P_BEFORE) tuples. The before kernel uses
TILE_P=TILE_P_BEFORE (< 128); the after kernel uses TILE_P=128.
"""

from __future__ import annotations

import pathlib

VARIANTS = [
    # (M, N, TILE_P_BEFORE) — M divisible by both TILE_P_BEFORE and 128
    (1024,  512,  16),
    (1024,  512,  32),  # v1 anchor
    (1024,  512,  64),
    (2048,  512,  32),  # off-axis: larger M
    (1024, 1024,  32),  # off-axis: larger N
]


_BEFORE_TEMPLATE = '''\
"""
Sweep variant — opt03 BEFORE: element-wise A+B with TILE_P={TILE_P} (<128).
M={M}, N={N}, TILE_P={TILE_P} → n_iters = M / TILE_P = {N_ITERS}.
"""
import nki
import nki.isa as nisa
import nki.language as nl

M = {M}
N = {N}
TILE_P = {TILE_P}


@nki.jit(platform_target="trn1")
def add_small_tiles(a, b):
    c = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    for i in nl.sequential_range(M // TILE_P):
        a_tile = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=a_tile, src=a[i * TILE_P:(i + 1) * TILE_P, 0:N])

        b_tile = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=b_tile, src=b[i * TILE_P:(i + 1) * TILE_P, 0:N])

        c_tile = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=c_tile, data1=a_tile, data2=b_tile, op=nl.add)

        nisa.dma_copy(dst=c[i * TILE_P:(i + 1) * TILE_P, 0:N], src=c_tile)

    return c
'''

_AFTER_TEMPLATE = '''\
"""
Sweep variant — opt03 AFTER: element-wise A+B with TILE_P=128 (full partition).
M={M}, N={N}, TILE_P=128 → n_iters = M / 128 = {N_ITERS}.
"""
import nki
import nki.isa as nisa
import nki.language as nl

M = {M}
N = {N}
TILE_P = 128


@nki.jit(platform_target="trn1")
def add_full_tiles(a, b):
    c = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    for i in nl.sequential_range(M // TILE_P):
        a_tile = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=a_tile, src=a[i * TILE_P:(i + 1) * TILE_P, 0:N])

        b_tile = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=b_tile, src=b[i * TILE_P:(i + 1) * TILE_P, 0:N])

        c_tile = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=c_tile, data1=a_tile, data2=b_tile, op=nl.add)

        nisa.dma_copy(dst=c[i * TILE_P:(i + 1) * TILE_P, 0:N], src=c_tile)

    return c
'''


def main() -> None:
    here = pathlib.Path(__file__).parent
    for M, N, TILE_P in VARIANTS:
        if M % TILE_P != 0 or M % 128 != 0:
            raise SystemExit(f"M={M} must be divisible by both TILE_P={TILE_P} and 128")
        ctx_b = dict(M=M, N=N, TILE_P=TILE_P, N_ITERS=M // TILE_P)
        ctx_a = dict(M=M, N=N, N_ITERS=M // 128)
        base = f"add_M{M}_N{N}_T{TILE_P}"
        (here / f"{base}_before.py").write_text(_BEFORE_TEMPLATE.format(**ctx_b))
        (here / f"{base}_after.py").write_text(_AFTER_TEMPLATE.format(**ctx_a))
        print(f"  wrote {base}_before.py + {base}_after.py")


if __name__ == "__main__":
    main()
