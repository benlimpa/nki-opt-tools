#!/usr/bin/env python3
"""
Generate parameterized HBM-accumulator before/after kernels for the opt01
sweep.

Usage:  python _generate.py

VARIANTS list: (M, TILE_M, N) tuples. The before kernel uses an HBM
accumulator with reload+store on every iteration; the after kernel keeps
the accumulator in SBUF and stores once at the end.
"""

from __future__ import annotations

import pathlib

VARIANTS = [
    # (M, TILE_M, N) — M divisible by TILE_M, TILE_M ≤ 128
    (1024, 128, 512),   # v1 anchor
    (2048, 128, 512),   # off-axis: larger M
    ( 512, 128, 512),   # off-axis: smaller M
    (1024, 128, 1024),  # off-axis: larger N (bigger accumulator)
    (1024,  64, 512),   # off-axis: smaller TILE_M (more iterations)
]


_BEFORE_TEMPLATE = '''\
"""
Sweep variant — opt01 BEFORE: column-wise reduction with HBM accumulator.
M={M}, TILE_M={TILE_M}, N={N} → n_tiles = M / TILE_M = {N_TILES}.
Generates {N_RT} unnecessary accumulator round-trips of [TILE_M, N] tile.
"""
import nki
import nki.isa as nisa
import nki.language as nl

M = {M}
TILE_M = {TILE_M}
N = {N}


@nki.jit(platform_target="trn1")
def reduce_sum_hbm_accumulator(x):
    y = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    acc_hbm = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.private_hbm)
    acc_init = nl.zeros((TILE_M, N), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=acc_hbm[0:TILE_M, 0:N], src=acc_init)

    for i in nl.sequential_range(M // TILE_M):
        x_tile = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tile, src=x[i * TILE_M:(i + 1) * TILE_M, 0:N])

        acc_sbuf = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=acc_sbuf, src=acc_hbm[0:TILE_M, 0:N])

        new_acc = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=new_acc, data1=acc_sbuf, data2=x_tile, op=nl.add)

        nisa.dma_copy(dst=acc_hbm[0:TILE_M, 0:N], src=new_acc)

    acc_final = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=acc_final, src=acc_hbm[0:TILE_M, 0:N])
    nisa.dma_copy(dst=y[0:TILE_M, 0:N], src=acc_final)

    return y
'''

_AFTER_TEMPLATE = '''\
"""
Sweep variant — opt01 AFTER: column-wise reduction with SBUF accumulator.
M={M}, TILE_M={TILE_M}, N={N} → n_tiles = M / TILE_M = {N_TILES}.
Single final store; zero unnecessary accumulator round-trips.
"""
import nki
import nki.isa as nisa
import nki.language as nl

M = {M}
TILE_M = {TILE_M}
N = {N}


@nki.jit(platform_target="trn1")
def reduce_sum_sbuf_accumulator(x):
    y = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    acc_sbuf = nl.zeros((TILE_M, N), dtype=nl.float32, buffer=nl.sbuf)

    for i in nl.sequential_range(M // TILE_M):
        x_tile = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tile, src=x[i * TILE_M:(i + 1) * TILE_M, 0:N])

        new_acc = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=new_acc, data1=acc_sbuf, data2=x_tile, op=nl.add)
        acc_sbuf = new_acc

    nisa.dma_copy(dst=y[0:TILE_M, 0:N], src=acc_sbuf)
    return y
'''


def main() -> None:
    here = pathlib.Path(__file__).parent
    for M, TILE_M, N in VARIANTS:
        if M % TILE_M != 0:
            raise SystemExit(f"M={M} must be divisible by TILE_M={TILE_M}")
        if TILE_M > 128:
            raise SystemExit(f"TILE_M={TILE_M} exceeds 128 partition limit")
        n_tiles = M // TILE_M
        ctx = dict(M=M, TILE_M=TILE_M, N=N, N_TILES=n_tiles, N_RT=2 * n_tiles)
        base = f"accum_M{M}_T{TILE_M}_N{N}"
        (here / f"{base}_before.py").write_text(_BEFORE_TEMPLATE.format(**ctx))
        (here / f"{base}_after.py").write_text(_AFTER_TEMPLATE.format(**ctx))
        print(f"  wrote {base}_before.py + {base}_after.py")


if __name__ == "__main__":
    main()
