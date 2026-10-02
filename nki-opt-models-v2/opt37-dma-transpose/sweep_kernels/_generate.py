#!/usr/bin/env python3
"""
Generate parameterized transpose before/after kernels for the opt37 sweep.

Usage:  python _generate.py

VARIANTS list: (D0, D3, NUM_ITERS) tuples. The transpose operates on a
[D0, 1, 1, D3] 4D tensor (D1=D2=1 to satisfy dma_transpose's quirks).
"""

from __future__ import annotations

import pathlib

VARIANTS = [
    # (D0, D3, NUM_ITERS) — D0 ≤ 128 (partition dim)
    (128, 128, 50),
    (128, 128, 100),   # v1 anchor
    (128, 128, 200),
    (128, 64,  100),   # off-axis: smaller free-dim
]


_BEFORE_TEMPLATE = '''\
"""
Sweep variant — opt37 BEFORE: nc_transpose (TensorE) in a loop.
D0={D0}, D1=1, D2=1, D3={D3}, NUM_ITERS={NUM_ITERS}.
"""
import nki
import nki.isa as nisa
import nki.language as nl

D0 = {D0}
D1 = 1
D2 = 1
D3 = {D3}
NUM_ITERS = {NUM_ITERS}


@nki.jit(platform_target="trn1")
def transpose_baseline(X):
    Y = nl.ndarray((D3, D1, D2, D0), dtype=nl.float32, buffer=nl.shared_hbm)

    x_tile = nl.ndarray((D0, D3), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=x_tile, src=X[:, 0, 0, :])

    y_psum = nl.zeros((D3, D0), dtype=nl.float32, buffer=nl.psum)

    for _ in nl.affine_range(NUM_ITERS):
        nisa.tensor_scalar(dst=y_psum, data=y_psum, op0=nl.multiply, operand0=0.0)
        nisa.nc_transpose(dst=y_psum, data=x_tile)

    y_sbuf = nl.ndarray((D3, D0), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=y_sbuf, op=nl.copy, data=y_psum)
    nisa.dma_copy(dst=Y[:, 0, 0, :], src=y_sbuf)
    return Y
'''

_AFTER_TEMPLATE = '''\
"""
Sweep variant — opt37 AFTER: dma_transpose (DMA Engine) in a loop.
D0={D0}, D1=1, D2=1, D3={D3}, NUM_ITERS={NUM_ITERS}.
"""
import nki
import nki.isa as nisa
import nki.language as nl

D0 = {D0}
D1 = 1
D2 = 1
D3 = {D3}
NUM_ITERS = {NUM_ITERS}


@nki.jit(platform_target="trn1")
def transpose_specialized(X):
    Y = nl.ndarray((D3, D1, D2, D0), dtype=nl.float32, buffer=nl.shared_hbm)
    y_tile = nl.ndarray((D3, D1, D2, D0), dtype=nl.float32, buffer=nl.sbuf)

    for _ in nl.affine_range(NUM_ITERS):
        nisa.dma_transpose(dst=y_tile, src=X)

    nisa.dma_copy(dst=Y, src=y_tile)
    return Y
'''


def main() -> None:
    here = pathlib.Path(__file__).parent
    for D0, D3, NUM_ITERS in VARIANTS:
        ctx = dict(D0=D0, D3=D3, NUM_ITERS=NUM_ITERS)
        base = f"transpose_D{D0}x{D3}_iters{NUM_ITERS}"
        (here / f"{base}_before.py").write_text(_BEFORE_TEMPLATE.format(**ctx))
        (here / f"{base}_after.py").write_text(_AFTER_TEMPLATE.format(**ctx))
        print(f"  wrote {base}_before.py + {base}_after.py")


if __name__ == "__main__":
    main()
