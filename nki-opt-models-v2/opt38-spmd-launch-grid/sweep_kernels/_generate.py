#!/usr/bin/env python3
"""
Generate parameterized element-wise add before/after kernels for the opt38
sweep.

Usage:  python _generate.py

VARIANTS list: (M, N, G) tuples. The before kernel is launched without an
SPMD grid (G=1 implicit); the after kernel is launched as kernel[G](...).
"""

from __future__ import annotations

import pathlib

TILE_P = 128

VARIANTS = [
    # (M, N, G) — M divisible by TILE_P; n_iters = M // TILE_P divisible by G
    (2048,  512, 2),
    (4096,  512, 2),
    (8192,  512, 2),
    (2048, 1024, 2),
    (4096,  512, 4),
]


_BEFORE_TEMPLATE = '''\
"""
Sweep variant — opt38 BEFORE: element-wise A+B without SPMD launch grid.
M={M}, N={N}, n_iters = M / TILE_P = {N_ITERS}; launched as kernel(args).
"""
import nki
import nki.isa as nisa
import nki.language as nl

M = {M}
N = {N}
TILE_P = 128


@nki.jit(platform_target="trn1")
def add_no_spmd(a, b):
    c = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    n_iters = M // TILE_P
    for i in nl.sequential_range(n_iters):
        a_tile = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=a_tile, src=a[i * TILE_P:(i + 1) * TILE_P, 0:N])

        b_tile = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=b_tile, src=b[i * TILE_P:(i + 1) * TILE_P, 0:N])

        c_tile = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=c_tile, data1=a_tile, data2=b_tile, op=nl.add)

        nisa.dma_copy(dst=c[i * TILE_P:(i + 1) * TILE_P, 0:N], src=c_tile)

    return c


def wrapper(a, b):
    return add_no_spmd(a, b)
'''

_AFTER_TEMPLATE = '''\
"""
Sweep variant — opt38 AFTER: element-wise A+B with SPMD launch grid G={G}.
M={M}, N={N}, n_iters = M / TILE_P = {N_ITERS}; launched as kernel[{G}](args).
"""
import nki
import nki.isa as nisa
import nki.language as nl

M = {M}
N = {N}
TILE_P = 128
GRID_SIZE = {G}


@nki.jit(platform_target="trn1")
def add_spmd(a, b):
    c = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    n_prgs = nl.num_programs(axes=0)
    prg_id = nl.program_id(axis=0)

    n_iters = M // TILE_P
    iters_per_prg = n_iters // n_prgs
    start = prg_id * iters_per_prg
    end = start + iters_per_prg

    for i in nl.sequential_range(start, end):
        a_tile = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=a_tile, src=a[i * TILE_P:(i + 1) * TILE_P, 0:N])

        b_tile = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=b_tile, src=b[i * TILE_P:(i + 1) * TILE_P, 0:N])

        c_tile = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=c_tile, data1=a_tile, data2=b_tile, op=nl.add)

        nisa.dma_copy(dst=c[i * TILE_P:(i + 1) * TILE_P, 0:N], src=c_tile)

    return c


def wrapper(a, b):
    return add_spmd[GRID_SIZE](a, b)
'''


def main() -> None:
    here = pathlib.Path(__file__).parent
    for M, N, G in VARIANTS:
        n_iters = M // TILE_P
        if M % TILE_P != 0 or n_iters % G != 0:
            raise SystemExit(f"M={M} must be divisible by TILE_P={TILE_P} and n_iters={n_iters} by G={G}")
        ctx_b = dict(M=M, N=N, N_ITERS=n_iters)
        ctx_a = dict(M=M, N=N, N_ITERS=n_iters, G=G)
        base = f"add_M{M}_N{N}_G{G}"
        (here / f"{base}_before.py").write_text(_BEFORE_TEMPLATE.format(**ctx_b))
        (here / f"{base}_after.py").write_text(_AFTER_TEMPLATE.format(**ctx_a))
        print(f"  wrote {base}_before.py + {base}_after.py")


if __name__ == "__main__":
    main()
