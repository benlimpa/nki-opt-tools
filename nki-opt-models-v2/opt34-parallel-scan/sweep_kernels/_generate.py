#!/usr/bin/env python3
"""
Generate parameterized 1D-cumsum before/after kernels for the opt34 sweep.

Usage:  python _generate.py

Materializes pairs of files of the form
    cumsum_N{N_TOTAL}_{before,after}.py
based on the VARIANTS list below. Re-run any time to refresh; existing files
are overwritten.

Both before and after take signature ``(X, L_tri)`` — the before kernel
ignores L_tri, but accepting it makes the shared oracle uniform.
"""

from __future__ import annotations

import pathlib

VARIANTS = [
    # N_total — must be divisible by P=128
    4096,    # TILE_N=32
    8192,    # TILE_N=64
    16384,   # TILE_N=128 (v1 anchor)
    32768,   # TILE_N=256
]

P = 128

_BEFORE_TEMPLATE = '''\
"""
Sweep variant — opt34 BEFORE: sequential row-by-row 1D cumsum.
N_TOTAL={N_TOTAL}, P={P}, TILE_N={TILE_N}.
"""
import nki
import nki.isa as nisa
import nki.language as nl

N_TOTAL = {N_TOTAL}
P = {P}
TILE_N = {TILE_N}


@nki.jit(platform_target="trn1")
def scan_sequential_rows(X, L_tri):
    # L_tri unused; kept for oracle parity with the AFTER kernel.
    X_2d = X.reshape((P, TILE_N))
    Y_2d = nl.ndarray((P, TILE_N), dtype=nl.float32, buffer=nl.shared_hbm)

    carry_sb = nl.zeros((P, 1), dtype=nl.float32, buffer=nl.sbuf, name="carry_sb")

    ones_sb = nl.zeros((P, TILE_N), dtype=nl.float32, buffer=nl.sbuf, name="ones_sb")
    nisa.tensor_scalar(dst=ones_sb, data=ones_sb, op0=nl.add, operand0=1.0)

    data_sb = nl.ndarray((P, TILE_N), dtype=nl.float32, buffer=nl.sbuf, name="data_sb")
    result_sb = nl.ndarray((P, TILE_N), dtype=nl.float32, buffer=nl.sbuf, name="result_sb")

    for i in nl.sequential_range(P):
        nisa.dma_copy(dst=data_sb[0:1, :], src=X_2d[i:i + 1, :])

        nisa.tensor_tensor_scan(
            dst=result_sb[0:1, :],
            data0=ones_sb[0:1, :],
            data1=data_sb[0:1, :],
            initial=carry_sb[0:1, :],
            op0=nl.multiply,
            op1=nl.add,
        )

        nisa.dma_copy(dst=Y_2d[i:i + 1, :], src=result_sb[0:1, :])

        nisa.tensor_copy(dst=carry_sb[0:1, :],
                         src=result_sb[0:1, TILE_N - 1:TILE_N])

    return Y_2d.reshape((N_TOTAL,))
'''

_AFTER_TEMPLATE = '''\
"""
Sweep variant — opt34 AFTER: parallel 1D cumsum via [P, TILE_N] reshape +
nc_matmul carry-over trick.
N_TOTAL={N_TOTAL}, P={P}, TILE_N={TILE_N}.
"""
import nki
import nki.isa as nisa
import nki.language as nl

N_TOTAL = {N_TOTAL}
P = {P}
TILE_N = {TILE_N}


@nki.jit(platform_target="trn1")
def scan_1d_parallel(X, L_tri):
    X_2d = X.reshape((P, TILE_N))
    Y_2d = nl.ndarray((P, TILE_N), dtype=nl.float32, buffer=nl.shared_hbm)

    data_sb = nl.ndarray((P, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=data_sb, src=X_2d)

    ones_sb = nl.zeros((P, TILE_N), dtype=nl.float32, buffer=nl.sbuf, name="ones_sb")
    nisa.tensor_scalar(dst=ones_sb, data=ones_sb, op0=nl.add, operand0=1.0)

    res_local_sb = nl.ndarray((P, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor_scan(
        dst=res_local_sb,
        data0=ones_sb,
        data1=data_sb,
        initial=0.0,
        op0=nl.multiply,
        op1=nl.add,
    )

    carries_sb = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=carries_sb, src=res_local_sb[:, TILE_N - 1:TILE_N])

    L_tri_sb = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=L_tri_sb, src=L_tri)

    offsets_psum = nl.zeros((P, 1), dtype=nl.float32, buffer=nl.psum,
                            name="offsets_psum")
    nisa.nc_matmul(dst=offsets_psum, stationary=L_tri_sb, moving=carries_sb)

    offsets_sb = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=offsets_sb, op=nl.copy, data=offsets_psum)

    final_res_sb = nl.ndarray((P, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
    for j in nl.affine_range(TILE_N):
        nisa.tensor_tensor(dst=final_res_sb[:, j:j + 1],
                           data1=res_local_sb[:, j:j + 1],
                           data2=offsets_sb,
                           op=nl.add)

    nisa.dma_copy(dst=Y_2d, src=final_res_sb)
    return Y_2d.reshape((N_TOTAL,))
'''


def main() -> None:
    here = pathlib.Path(__file__).parent
    for N_TOTAL in VARIANTS:
        TILE_N = N_TOTAL // P
        ctx = dict(N_TOTAL=N_TOTAL, P=P, TILE_N=TILE_N)
        base = f"cumsum_N{N_TOTAL}"
        (here / f"{base}_before.py").write_text(_BEFORE_TEMPLATE.format(**ctx))
        (here / f"{base}_after.py").write_text(_AFTER_TEMPLATE.format(**ctx))
        print(f"  wrote {base}_before.py + {base}_after.py")


if __name__ == "__main__":
    main()
