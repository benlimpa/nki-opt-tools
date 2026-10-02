#!/usr/bin/env python3
"""
Generate parameterized polynomial-pipeline before/after kernels for opt24.

Usage:  python _generate.py

VARIANTS list: (P, N, TILE_N) tuples. The before kernel allocates SBUF
tiles in fp32 (forcing implicit bf16↔fp32 casts on every dma_copy);
the after kernel uses bf16 SBUF tiles natively.

Both kernels compute y = 0.79788 * (x + 0.044715 * x^3).
"""

from __future__ import annotations

import pathlib

VARIANTS = [
    # (P, N, TILE_N) — N divisible by TILE_N, P ≤ 128
    (128,  32768, 512),
    (128,  65536, 512),
    (128, 131072, 512),  # v1 anchor
    ( 64, 131072, 512),  # off-axis: smaller P
    (128,  16384, 512),  # off-axis: smaller N
]


_BEFORE_TEMPLATE = '''\
"""
Sweep variant — opt24 BEFORE: fp32 internal SBUF tiles (implicit casts).
P={P}, N={N}, TILE_N={TILE_N}.
"""
import nki
import nki.isa as nisa
import nki.language as nl

P = {P}
N = {N}
TILE_N = {TILE_N}


@nki.jit(platform_target="trn1")
def poly_float32(x):
    y = nl.ndarray((P, N), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    for n in nl.sequential_range(N // TILE_N):
        x_tile = nl.ndarray((P, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tile, src=x[0:P, n*TILE_N:(n+1)*TILE_N])

        x2 = nl.ndarray((P, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=x2, data1=x_tile, data2=x_tile, op=nl.multiply)

        x3 = nl.ndarray((P, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=x3, data1=x2, data2=x_tile, op=nl.multiply)

        term1 = nl.ndarray((P, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=term1, data=x3, op0=nl.multiply, operand0=0.044715)

        res = nl.ndarray((P, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=res, data1=x_tile, data2=term1, op=nl.add)

        res2 = nl.ndarray((P, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=res2, data=res, op0=nl.multiply, operand0=0.79788)

        nisa.dma_copy(dst=y[0:P, n*TILE_N:(n+1)*TILE_N], src=res2)

    return y
'''

_AFTER_TEMPLATE = '''\
"""
Sweep variant — opt24 AFTER: bf16 internal SBUF tiles (no implicit casts).
P={P}, N={N}, TILE_N={TILE_N}.
"""
import nki
import nki.isa as nisa
import nki.language as nl

P = {P}
N = {N}
TILE_N = {TILE_N}


@nki.jit(platform_target="trn1")
def poly_bfloat16(x):
    y = nl.ndarray((P, N), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    for n in nl.sequential_range(N // TILE_N):
        x_tile = nl.ndarray((P, TILE_N), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tile, src=x[0:P, n*TILE_N:(n+1)*TILE_N])

        x2 = nl.ndarray((P, TILE_N), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=x2, data1=x_tile, data2=x_tile, op=nl.multiply)

        x3 = nl.ndarray((P, TILE_N), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=x3, data1=x2, data2=x_tile, op=nl.multiply)

        term1 = nl.ndarray((P, TILE_N), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=term1, data=x3, op0=nl.multiply, operand0=0.044715)

        res = nl.ndarray((P, TILE_N), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=res, data1=x_tile, data2=term1, op=nl.add)

        res2 = nl.ndarray((P, TILE_N), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=res2, data=res, op0=nl.multiply, operand0=0.79788)

        nisa.dma_copy(dst=y[0:P, n*TILE_N:(n+1)*TILE_N], src=res2)

    return y
'''


def main() -> None:
    here = pathlib.Path(__file__).parent
    for P, N, TILE_N in VARIANTS:
        if N % TILE_N != 0:
            raise SystemExit(f"N={N} must be divisible by TILE_N={TILE_N}")
        ctx = dict(P=P, N=N, TILE_N=TILE_N)
        base = f"poly_P{P}_N{N}"
        (here / f"{base}_before.py").write_text(_BEFORE_TEMPLATE.format(**ctx))
        (here / f"{base}_after.py").write_text(_AFTER_TEMPLATE.format(**ctx))
        print(f"  wrote {base}_before.py + {base}_after.py")


if __name__ == "__main__":
    main()
