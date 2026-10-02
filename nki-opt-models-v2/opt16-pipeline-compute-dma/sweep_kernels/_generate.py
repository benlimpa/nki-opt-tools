#!/usr/bin/env python3
"""
Generate parameterized matmul before/after kernels for the opt16 sweep.

Usage:  python _generate.py

Materializes pairs of files of the form
    mm_K{K}_M{M}_N{N}_{dtype}_{before,after}.py
based on the VARIANTS list below. Re-run any time to refresh; existing files
are overwritten.

Each kernel defines `matmul_serial` (before) or `matmul_pipelined` (after) as
the last function — the wrapper-name extractor will pick it up automatically.
The kernels also expose top-level constants ``M``, ``K``, ``N``, ``DTYPE`` that
the shared ``mm_oracle.py`` reads.
"""

from __future__ import annotations

import pathlib

VARIANTS = [
    # (M, K, N, TILE_M, TILE_K, TILE_N, dtype)
    (512,  128, 512, 128, 128, 512, "fp32"),
    (1024, 128, 512, 128, 128, 512, "fp32"),
    (2048, 128, 512, 128, 128, 512, "fp32"),
    (1024, 128, 512, 128, 128, 512, "bf16"),  # off-axis: dtype
]

_NL_DTYPE = {"fp32": "nl.float32", "bf16": "nl.bfloat16"}
_TORCH_DTYPE = {"fp32": "torch.float32", "bf16": "torch.bfloat16"}

_BEFORE_TEMPLATE = '''\
"""
Sweep variant — opt16 BEFORE: serial load/compute/store per M-tile.
M={M}, K={K}, N={N}, TILE_M={TILE_M}, dtype={dtype}.
"""
import nki
import nki.isa as nisa
import nki.language as nl

M = {M}
K = {K}
N = {N}
TILE_M = {TILE_M}
TILE_K = {TILE_K}
TILE_N = {TILE_N}
DTYPE = "{dtype}"
_NL_DTYPE = {nl_dtype}


@nki.jit(platform_target="trn1")
def matmul_serial(lhsT, rhs):
    result = nl.ndarray((M, N), dtype=_NL_DTYPE, buffer=nl.shared_hbm)

    rhs_tile = nl.ndarray((TILE_K, TILE_N), dtype=_NL_DTYPE, buffer=nl.sbuf)
    nisa.dma_copy(dst=rhs_tile, src=rhs[0:TILE_K, 0:TILE_N])

    lhsT_tile = nl.ndarray((TILE_K, TILE_M), dtype=_NL_DTYPE, buffer=nl.sbuf)
    res_sbuf = nl.ndarray((TILE_M, TILE_N), dtype=_NL_DTYPE, buffer=nl.sbuf)

    for m in nl.sequential_range(M // TILE_M):
        nisa.dma_copy(dst=lhsT_tile,
                      src=lhsT[0:TILE_K, m * TILE_M:(m + 1) * TILE_M])

        res_psum = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=res_psum, stationary=lhsT_tile, moving=rhs_tile)
        nisa.activation(dst=res_sbuf, op=nl.copy, data=res_psum)

        nisa.dma_copy(dst=result[m * TILE_M:(m + 1) * TILE_M, 0:TILE_N],
                      src=res_sbuf)

    return result
'''

_AFTER_TEMPLATE = '''\
"""
Sweep variant — opt16 AFTER: 3-stage pipeline (input + output ping-pong).
M={M}, K={K}, N={N}, TILE_M={TILE_M}, dtype={dtype}.
Requires M / TILE_M >= 3 (prologue+epilogue fill the pipeline).
"""
import nki
import nki.isa as nisa
import nki.language as nl

M = {M}
K = {K}
N = {N}
TILE_M = {TILE_M}
TILE_K = {TILE_K}
TILE_N = {TILE_N}
DTYPE = "{dtype}"
_NL_DTYPE = {nl_dtype}


@nki.jit(platform_target="trn1")
def matmul_pipelined(lhsT, rhs):
    result = nl.ndarray((M, N), dtype=_NL_DTYPE, buffer=nl.shared_hbm)

    rhs_tile = nl.ndarray((TILE_K, TILE_N), dtype=_NL_DTYPE, buffer=nl.sbuf)
    nisa.dma_copy(dst=rhs_tile, src=rhs[0:TILE_K, 0:TILE_N])

    NUM_TILES = M // TILE_M

    input_bufs = nl.ndarray((TILE_K, 2, TILE_M), dtype=_NL_DTYPE, buffer=nl.sbuf)
    output_bufs = nl.ndarray((TILE_M, 2, TILE_N), dtype=_NL_DTYPE, buffer=nl.sbuf)

    # Prologue
    nisa.dma_copy(dst=input_bufs[0:TILE_K, 0, 0:TILE_M],
                  src=lhsT[0:TILE_K, 0:TILE_M])
    nisa.dma_copy(dst=input_bufs[0:TILE_K, 1, 0:TILE_M],
                  src=lhsT[0:TILE_K, TILE_M:2 * TILE_M])
    res_psum = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=res_psum,
                   stationary=input_bufs[0:TILE_K, 0, 0:TILE_M],
                   moving=rhs_tile)
    nisa.activation(dst=output_bufs[0:TILE_M, 0, 0:TILE_N],
                    op=nl.copy, data=res_psum)

    # Steady state
    for m in nl.sequential_range(1, NUM_TILES - 1):
        nisa.dma_copy(dst=input_bufs[0:TILE_K, (m + 1) % 2, 0:TILE_M],
                      src=lhsT[0:TILE_K, (m + 1) * TILE_M:(m + 2) * TILE_M])
        nisa.dma_copy(dst=result[(m - 1) * TILE_M:m * TILE_M, 0:TILE_N],
                      src=output_bufs[0:TILE_M, (m - 1) % 2, 0:TILE_N])

        res_psum = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=res_psum,
                       stationary=input_bufs[0:TILE_K, m % 2, 0:TILE_M],
                       moving=rhs_tile)
        nisa.activation(dst=output_bufs[0:TILE_M, m % 2, 0:TILE_N],
                        op=nl.copy, data=res_psum)

    # Epilogue
    last = NUM_TILES - 1
    nisa.dma_copy(dst=result[(last - 1) * TILE_M:last * TILE_M, 0:TILE_N],
                  src=output_bufs[0:TILE_M, (last - 1) % 2, 0:TILE_N])
    res_psum = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=res_psum,
                   stationary=input_bufs[0:TILE_K, last % 2, 0:TILE_M],
                   moving=rhs_tile)
    nisa.activation(dst=output_bufs[0:TILE_M, last % 2, 0:TILE_N],
                    op=nl.copy, data=res_psum)
    nisa.dma_copy(dst=result[last * TILE_M:(last + 1) * TILE_M, 0:TILE_N],
                  src=output_bufs[0:TILE_M, last % 2, 0:TILE_N])

    return result
'''


def _name(M: int, K: int, N: int, dtype: str) -> str:
    return f"mm_K{K}_M{M}_N{N}_{dtype}"


def main() -> None:
    here = pathlib.Path(__file__).parent
    for M, K, N, TILE_M, TILE_K, TILE_N, dtype in VARIANTS:
        ctx = dict(M=M, K=K, N=N, TILE_M=TILE_M, TILE_K=TILE_K, TILE_N=TILE_N,
                   dtype=dtype, nl_dtype=_NL_DTYPE[dtype])
        base = _name(M, K, N, dtype)
        (here / f"{base}_before.py").write_text(_BEFORE_TEMPLATE.format(**ctx))
        (here / f"{base}_after.py").write_text(_AFTER_TEMPLATE.format(**ctx))
        print(f"  wrote {base}_before.py + {base}_after.py")


if __name__ == "__main__":
    main()
