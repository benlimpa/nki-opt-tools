#!/usr/bin/env python3
"""
Generate parameterized minmax-norm before/after kernels for the opt31 sweep.

Usage:  python _generate.py

Materializes pairs of files of the form
    minmax_M{M}_N{N}_T{TILE_N}_{before,after}.py
based on the VARIANTS list below. Re-run any time to refresh; existing files
are overwritten.

Each kernel exposes top-level constants ``M``, ``N``, ``TILE_P``,
``TILE_N_BEFORE``, ``TILE_N_AFTER`` that the shared ``minmax_oracle.py``
reads. The before kernel's actual TILE_N is ``TILE_N_BEFORE``; the after
kernel uses ``TILE_N_AFTER`` (typically equal to N, i.e. one block).
"""

from __future__ import annotations

import pathlib

# (M, N, TILE_N_BEFORE) — TILE_N_AFTER is always N (full free axis).
# Constraint: N % TILE_N_BEFORE == 0.
VARIANTS = [
    # Vary tile_ratio at N=512 (M=1024 fixed)
    (1024, 512, 32),    # tile_ratio=16  ← v1 anchor
    (1024, 512, 64),    # tile_ratio=8
    (1024, 512, 128),   # tile_ratio=4
    # Off-axis: smaller tile_ratio (still beneficial?)
    (1024, 512, 256),   # tile_ratio=2
    # Off-axis: vary N
    (1024, 1024, 64),   # tile_ratio=16, N doubled
    (512, 256, 32),     # tile_ratio=8,  smaller M and N
]

TILE_P = 128


_BEFORE_TEMPLATE = '''\
"""
Sweep variant — opt31 BEFORE: small TILE_N min-max norm, two-pass with reload.
M={M}, N={N}, TILE_P={TILE_P}, TILE_N={TILE_N}.
"""
import nki
import nki.isa as nisa
import nki.language as nl

M = {M}
N = {N}
TILE_P = {TILE_P}
TILE_N_BEFORE = {TILE_N}
TILE_N_AFTER = {TILE_N_AFTER}
EPS = 1e-8


@nki.jit(platform_target="trn1")
def minmax_norm(x):
    TILE_N = TILE_N_BEFORE
    y = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    for m in nl.affine_range(M // TILE_P):
        x0 = nl.ndarray((TILE_P, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x0,
                      src=x[m * TILE_P:(m + 1) * TILE_P, 0:TILE_N])

        min_val = nl.ndarray((TILE_P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=min_val, data=x0,
                           op=nl.minimum, axis=1, keepdims=True)
        max_val = nl.ndarray((TILE_P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=max_val, data=x0,
                           op=nl.maximum, axis=1, keepdims=True)

        for n in nl.sequential_range(N // TILE_N - 1):
            x_tile = nl.ndarray((TILE_P, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=x_tile,
                          src=x[m * TILE_P:(m + 1) * TILE_P,
                                (n + 1) * TILE_N:(n + 2) * TILE_N])

            tile_min = nl.ndarray((TILE_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=tile_min, data=x_tile,
                               op=nl.minimum, axis=1, keepdims=True)
            nisa.tensor_tensor(dst=min_val, data1=min_val, data2=tile_min,
                               op=nl.minimum)

            tile_max = nl.ndarray((TILE_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=tile_max, data=x_tile,
                               op=nl.maximum, axis=1, keepdims=True)
            nisa.tensor_tensor(dst=max_val, data1=max_val, data2=tile_max,
                               op=nl.maximum)

        range_val = nl.ndarray((TILE_P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=range_val, data=max_val,
                           op0=nl.subtract, operand0=min_val)
        range_eps = nl.ndarray((TILE_P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=range_eps, data=range_val,
                           op0=nl.add, operand0=EPS)
        inv_range = nl.ndarray((TILE_P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=inv_range, data=range_eps, op=nl.reciprocal)

        for n in nl.affine_range(N // TILE_N):
            x_tile2 = nl.ndarray((TILE_P, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=x_tile2,
                          src=x[m * TILE_P:(m + 1) * TILE_P,
                                n * TILE_N:(n + 1) * TILE_N])

            x_shifted = nl.ndarray((TILE_P, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=x_shifted, data=x_tile2,
                               op0=nl.subtract, operand0=min_val)

            y_tile = nl.ndarray((TILE_P, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=y_tile, data=x_shifted,
                               op0=nl.multiply, operand0=inv_range)

            nisa.dma_copy(dst=y[m * TILE_P:(m + 1) * TILE_P,
                                n * TILE_N:(n + 1) * TILE_N],
                          src=y_tile)

    return y
'''

_AFTER_TEMPLATE = '''\
"""
Sweep variant — opt31 AFTER: full free-dim block, single-pass minmax norm.
M={M}, N={N}, TILE_P={TILE_P}, TILE_N={TILE_N_AFTER} (= N).
"""
import nki
import nki.isa as nisa
import nki.language as nl

M = {M}
N = {N}
TILE_P = {TILE_P}
TILE_N_BEFORE = {TILE_N}
TILE_N_AFTER = {TILE_N_AFTER}
EPS = 1e-8


@nki.jit(platform_target="trn1")
def minmax_norm(x):
    TILE_N = TILE_N_AFTER
    y = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    for m in nl.affine_range(M // TILE_P):
        x_block = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_block,
                      src=x[m * TILE_P:(m + 1) * TILE_P, 0:N])

        min_val = nl.ndarray((TILE_P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=min_val, data=x_block,
                           op=nl.minimum, axis=1, keepdims=True)
        max_val = nl.ndarray((TILE_P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=max_val, data=x_block,
                           op=nl.maximum, axis=1, keepdims=True)

        range_val = nl.ndarray((TILE_P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=range_val, data=max_val,
                           op0=nl.subtract, operand0=min_val)
        range_eps = nl.ndarray((TILE_P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=range_eps, data=range_val,
                           op0=nl.add, operand0=EPS)
        inv_range = nl.ndarray((TILE_P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=inv_range, data=range_eps, op=nl.reciprocal)

        x_shifted = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=x_shifted, data=x_block,
                           op0=nl.subtract, operand0=min_val)
        y_block = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=y_block, data=x_shifted,
                           op0=nl.multiply, operand0=inv_range)

        nisa.dma_copy(dst=y[m * TILE_P:(m + 1) * TILE_P, 0:N],
                      src=y_block)

    return y
'''


def main() -> None:
    here = pathlib.Path(__file__).parent
    for M, N, TILE_N in VARIANTS:
        ctx = dict(M=M, N=N, TILE_P=TILE_P, TILE_N=TILE_N, TILE_N_AFTER=N)
        base = f"minmax_M{M}_N{N}_T{TILE_N}"
        (here / f"{base}_before.py").write_text(_BEFORE_TEMPLATE.format(**ctx))
        (here / f"{base}_after.py").write_text(_AFTER_TEMPLATE.format(**ctx))
        print(f"  wrote {base}_before.py + {base}_after.py")


if __name__ == "__main__":
    main()
