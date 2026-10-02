#!/usr/bin/env python3
"""
Generate parameterized softmax before/after kernels for the opt10 sweep.

Usage:  python _generate.py

VARIANTS list: (P, N_total, TILE_N) tuples. Both kernels share P, N_total,
TILE_N globals; n_tiles = N_total // TILE_N.
"""

from __future__ import annotations

import pathlib

VARIANTS = [
    # (P, N_total, TILE_N) — n_tiles = N_total // TILE_N
    (128,  512, 128),  # v1 anchor: n_tiles=4
    (128, 1024, 128),  # n_tiles=8
    (128, 2048, 128),  # n_tiles=16
    (128,  512,  64),  # n_tiles=8 (smaller tile)
    (128, 1024, 256),  # n_tiles=4 (larger tile)
]


_BEFORE_TEMPLATE = '''\
"""
Sweep variant — opt10 BEFORE: softmax with premature exp_hbm intermediate.
P={P}, N_total={N_total}, TILE_N={TILE_N} → n_tiles={N_TILES}.
"""
import nki
import nki.isa as nisa
import nki.language as nl

P = {P}
N_total = {N_total}
TILE_N = {TILE_N}


@nki.jit(platform_target="trn1")
def softmax_hbm_intermediate(x):
    y = nl.ndarray((P, N_total), dtype=nl.float32, buffer=nl.shared_hbm)

    max_val = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=max_val, value=-float("inf"))

    for n in nl.sequential_range(N_total // TILE_N):
        x_tile = nl.ndarray((P, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tile, src=x[0:P, n * TILE_N:(n + 1) * TILE_N])

        tile_max = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=tile_max, op=nl.maximum, data=x_tile, axis=1, keepdims=True)

        new_max = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=new_max, data1=max_val, data2=tile_max, op=nl.maximum)
        max_val = new_max

    neg_max = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=neg_max, op=nl.copy, data=max_val, scale=-1.0)

    exp_hbm = nl.ndarray((P, N_total), dtype=nl.float32, buffer=nl.private_hbm)
    exp_sum = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=exp_sum, value=0.0)

    for n in nl.sequential_range(N_total // TILE_N):
        x_tile = nl.ndarray((P, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tile, src=x[0:P, n * TILE_N:(n + 1) * TILE_N])

        exp_tile = nl.ndarray((P, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=exp_tile, op=nl.exp, data=x_tile, bias=neg_max)

        # PREMATURE STORE: write exp to HBM only to reload later
        nisa.dma_copy(dst=exp_hbm[0:P, n * TILE_N:(n + 1) * TILE_N], src=exp_tile)

        tile_sum = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=tile_sum, op=nl.add, data=exp_tile, axis=1, keepdims=True)

        new_sum = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=new_sum, data1=exp_sum, data2=tile_sum, op=nl.add)
        exp_sum = new_sum

    inv_sum = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=inv_sum, op=nl.reciprocal, data=exp_sum)

    for n in nl.affine_range(N_total // TILE_N):
        exp_tile = nl.ndarray((P, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=exp_tile, src=exp_hbm[0:P, n * TILE_N:(n + 1) * TILE_N])

        out_tile = nl.ndarray((P, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=out_tile, op=nl.copy, data=exp_tile, scale=inv_sum)

        nisa.dma_copy(dst=y[0:P, n * TILE_N:(n + 1) * TILE_N], src=out_tile)

    return y
'''

_AFTER_TEMPLATE = '''\
"""
Sweep variant — opt10 AFTER: softmax with delayed division (no exp_hbm).
P={P}, N_total={N_total}, TILE_N={TILE_N} → n_tiles={N_TILES}.
"""
import nki
import nki.isa as nisa
import nki.language as nl

P = {P}
N_total = {N_total}
TILE_N = {TILE_N}


@nki.jit(platform_target="trn1")
def softmax_no_intermediate(x):
    y = nl.ndarray((P, N_total), dtype=nl.float32, buffer=nl.shared_hbm)

    max_val = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=max_val, value=-float("inf"))

    for n in nl.sequential_range(N_total // TILE_N):
        x_tile = nl.ndarray((P, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tile, src=x[0:P, n * TILE_N:(n + 1) * TILE_N])

        tile_max = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=tile_max, op=nl.maximum, data=x_tile, axis=1, keepdims=True)

        new_max = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=new_max, data1=max_val, data2=tile_max, op=nl.maximum)
        max_val = new_max

    neg_max = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=neg_max, op=nl.copy, data=max_val, scale=-1.0)

    exp_sum = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=exp_sum, value=0.0)

    for n in nl.sequential_range(N_total // TILE_N):
        x_tile = nl.ndarray((P, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tile, src=x[0:P, n * TILE_N:(n + 1) * TILE_N])

        exp_tile = nl.ndarray((P, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=exp_tile, op=nl.exp, data=x_tile, bias=neg_max)

        # No store; exp_tile discarded.
        tile_sum = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=tile_sum, op=nl.add, data=exp_tile, axis=1, keepdims=True)

        new_sum = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=new_sum, data1=exp_sum, data2=tile_sum, op=nl.add)
        exp_sum = new_sum

    inv_sum = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=inv_sum, op=nl.reciprocal, data=exp_sum)

    for n in nl.affine_range(N_total // TILE_N):
        x_tile = nl.ndarray((P, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tile, src=x[0:P, n * TILE_N:(n + 1) * TILE_N])

        exp_tile = nl.ndarray((P, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=exp_tile, op=nl.exp, data=x_tile, bias=neg_max)

        out_tile = nl.ndarray((P, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=out_tile, op=nl.copy, data=exp_tile, scale=inv_sum)

        nisa.dma_copy(dst=y[0:P, n * TILE_N:(n + 1) * TILE_N], src=out_tile)

    return y
'''


def main() -> None:
    here = pathlib.Path(__file__).parent
    for P, N_total, TILE_N in VARIANTS:
        if N_total % TILE_N != 0:
            raise SystemExit(f"N_total={N_total} must be divisible by TILE_N={TILE_N}")
        ctx = dict(P=P, N_total=N_total, TILE_N=TILE_N, N_TILES=N_total // TILE_N)
        base = f"sm_P{P}_N{N_total}_T{TILE_N}"
        (here / f"{base}_before.py").write_text(_BEFORE_TEMPLATE.format(**ctx))
        (here / f"{base}_after.py").write_text(_AFTER_TEMPLATE.format(**ctx))
        print(f"  wrote {base}_before.py + {base}_after.py")


if __name__ == "__main__":
    main()
