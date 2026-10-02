#!/usr/bin/env python3
"""
Generate parameterized tanh-fused-scale-bias kernels for the opt35 sweep.

Usage:  python _generate.py

Materializes pairs of files of the form
    tanh_N{N}_ops{ops}_iters{iters}_{before,after}.py
based on the VARIANTS list below. Re-run any time to refresh; existing files
are overwritten.

Each kernel exposes top-level constants ``P``, ``N``, ``NUM_ITERS``, ``OPS``
that the shared ``tanh_oracle.py`` reads. ``OPS`` is 1 (scale-only) or 2
(scale + bias).
"""

from __future__ import annotations

import pathlib

VARIANTS = [
    # (N, OPS, NUM_ITERS) — 3 knobs, ≥2 distinct values per knob
    (128, 2, 50),    # vary N (small)
    (256, 2, 50),    # vary N (mid)
    (512, 2, 50),    # vary N (large) — closest to v1 anchor
    (256, 1, 50),    # vary fused_ops down
    (256, 2, 25),    # vary num_outer_iters down
    (256, 2, 100),   # vary num_outer_iters up
]

P = 128  # partition dim is fixed at 128 (free axis = N)


_BEFORE_TEMPLATE_OPS2 = '''\
"""
Sweep variant — opt35 BEFORE (manual broadcast loop, scale + bias).
P={P}, N={N}, OPS={OPS}, NUM_ITERS={NUM_ITERS}.
"""
import nki
import nki.isa as nisa
import nki.language as nl

P = {P}
N = {N}
OPS = {OPS}
NUM_ITERS = {NUM_ITERS}


@nki.jit(platform_target="trn1")
def tanh_baseline(X, scale, bias):
    Y = nl.ndarray((P, N), dtype=nl.float32, buffer=nl.shared_hbm)

    x_tile = nl.ndarray((P, N), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=x_tile, src=X)

    s_tile = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=s_tile, src=scale)

    b_tile = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=b_tile, src=bias)

    res_tile = nl.ndarray((P, N), dtype=nl.float32, buffer=nl.sbuf)

    for _ in nl.affine_range(NUM_ITERS):
        tmp1 = nl.ndarray((P, N), dtype=nl.float32, buffer=nl.sbuf)
        for j in nl.affine_range(N):
            nisa.tensor_tensor(dst=tmp1[:, j:j+1], data1=x_tile[:, j:j+1],
                               data2=s_tile, op=nl.multiply)

        tmp2 = nl.ndarray((P, N), dtype=nl.float32, buffer=nl.sbuf)
        for j in nl.affine_range(N):
            nisa.tensor_tensor(dst=tmp2[:, j:j+1], data1=tmp1[:, j:j+1],
                               data2=b_tile, op=nl.add)

        nisa.activation(dst=res_tile, data=tmp2, op=nl.tanh)

    nisa.dma_copy(dst=Y, src=res_tile)
    return Y
'''

_BEFORE_TEMPLATE_OPS1 = '''\
"""
Sweep variant — opt35 BEFORE (manual broadcast loop, scale-only).
P={P}, N={N}, OPS={OPS}, NUM_ITERS={NUM_ITERS}.
"""
import nki
import nki.isa as nisa
import nki.language as nl

P = {P}
N = {N}
OPS = {OPS}
NUM_ITERS = {NUM_ITERS}


@nki.jit(platform_target="trn1")
def tanh_baseline(X, scale, bias):
    # bias is unused in OPS=1 variant but kept in the signature for oracle parity
    Y = nl.ndarray((P, N), dtype=nl.float32, buffer=nl.shared_hbm)

    x_tile = nl.ndarray((P, N), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=x_tile, src=X)

    s_tile = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=s_tile, src=scale)

    res_tile = nl.ndarray((P, N), dtype=nl.float32, buffer=nl.sbuf)

    for _ in nl.affine_range(NUM_ITERS):
        tmp1 = nl.ndarray((P, N), dtype=nl.float32, buffer=nl.sbuf)
        for j in nl.affine_range(N):
            nisa.tensor_tensor(dst=tmp1[:, j:j+1], data1=x_tile[:, j:j+1],
                               data2=s_tile, op=nl.multiply)

        nisa.activation(dst=res_tile, data=tmp1, op=nl.tanh)

    nisa.dma_copy(dst=Y, src=res_tile)
    return Y
'''

_AFTER_TEMPLATE_OPS2 = '''\
"""
Sweep variant — opt35 AFTER (fused nisa.activation, scale + bias).
P={P}, N={N}, OPS={OPS}, NUM_ITERS={NUM_ITERS}.
"""
import nki
import nki.isa as nisa
import nki.language as nl

P = {P}
N = {N}
OPS = {OPS}
NUM_ITERS = {NUM_ITERS}


@nki.jit(platform_target="trn1")
def tanh_specialized(X, scale, bias):
    Y = nl.ndarray((P, N), dtype=nl.float32, buffer=nl.shared_hbm)

    x_tile = nl.ndarray((P, N), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=x_tile, src=X)

    s_tile = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=s_tile, src=scale)

    b_tile = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=b_tile, src=bias)

    res_tile = nl.ndarray((P, N), dtype=nl.float32, buffer=nl.sbuf)

    for _ in nl.affine_range(NUM_ITERS):
        nisa.activation(dst=res_tile, op=nl.tanh, data=x_tile,
                        scale=s_tile, bias=b_tile)

    nisa.dma_copy(dst=Y, src=res_tile)
    return Y
'''

_AFTER_TEMPLATE_OPS1 = '''\
"""
Sweep variant — opt35 AFTER (fused nisa.activation, scale-only).
P={P}, N={N}, OPS={OPS}, NUM_ITERS={NUM_ITERS}.
"""
import nki
import nki.isa as nisa
import nki.language as nl

P = {P}
N = {N}
OPS = {OPS}
NUM_ITERS = {NUM_ITERS}


@nki.jit(platform_target="trn1")
def tanh_specialized(X, scale, bias):
    Y = nl.ndarray((P, N), dtype=nl.float32, buffer=nl.shared_hbm)

    x_tile = nl.ndarray((P, N), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=x_tile, src=X)

    s_tile = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=s_tile, src=scale)

    res_tile = nl.ndarray((P, N), dtype=nl.float32, buffer=nl.sbuf)

    for _ in nl.affine_range(NUM_ITERS):
        nisa.activation(dst=res_tile, op=nl.tanh, data=x_tile, scale=s_tile)

    nisa.dma_copy(dst=Y, src=res_tile)
    return Y
'''


def _name(N: int, ops: int, iters: int) -> str:
    return f"tanh_N{N}_ops{ops}_iters{iters}"


def main() -> None:
    here = pathlib.Path(__file__).parent
    for N, OPS, NUM_ITERS in VARIANTS:
        ctx = dict(P=P, N=N, OPS=OPS, NUM_ITERS=NUM_ITERS)
        before_tmpl = _BEFORE_TEMPLATE_OPS2 if OPS == 2 else _BEFORE_TEMPLATE_OPS1
        after_tmpl = _AFTER_TEMPLATE_OPS2 if OPS == 2 else _AFTER_TEMPLATE_OPS1
        base = _name(N, OPS, NUM_ITERS)
        (here / f"{base}_before.py").write_text(before_tmpl.format(**ctx))
        (here / f"{base}_after.py").write_text(after_tmpl.format(**ctx))
        print(f"  wrote {base}_before.py + {base}_after.py")


if __name__ == "__main__":
    main()
