#!/usr/bin/env python3
"""
Generate parameterized scale-with-padding before/after kernels for opt06.

Usage:  python _generate.py

VARIANTS list: (M, N, TILE_P, VALID_TILES) tuples. The before kernel
iterates total_tiles=M//TILE_P; the after kernel stops at valid_tiles.
"""

from __future__ import annotations

import pathlib

VARIANTS = [
    # (M, N, TILE_P, VALID_TILES) — total_tiles = M // TILE_P
    (1024, 512, 128, 4),  # v1 anchor: total=8, valid=4 (50% padding)
    (1024, 512, 128, 2),  # heavier padding: total=8, valid=2 (75%)
    (1024, 512, 128, 6),  # lighter padding: total=8, valid=6 (25%)
    (2048, 512, 128, 8),  # off-axis larger M: total=16, valid=8 (50%)
    (1024, 1024, 128, 4), # off-axis larger N: total=8, valid=4 (50%)
]


_BEFORE_TEMPLATE = '''\
"""
Sweep variant — opt06 BEFORE: scale all tiles incl. padding.
M={M}, N={N}, TILE_P={TILE_P}, VALID_TILES={VALID_TILES}
total_tiles = M // TILE_P = {TOTAL_TILES} (loop iterates all of them).
"""
import nki
import nki.isa as nisa
import nki.language as nl

M = {M}
N = {N}
TILE_P = {TILE_P}
TOTAL_TILES = {TOTAL_TILES}
VALID_TILES = {VALID_TILES}


@nki.jit(platform_target="trn1")
def scale_all_tiles(x):
    y = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    for i in nl.sequential_range(TOTAL_TILES):
        x_tile = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tile, src=x[i * TILE_P:(i + 1) * TILE_P, 0:N])

        result = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=result, data=x_tile, op0=nl.multiply, operand0=2.0)

        nisa.dma_copy(dst=y[i * TILE_P:(i + 1) * TILE_P, 0:N], src=result)

    return y
'''

_AFTER_TEMPLATE = '''\
"""
Sweep variant — opt06 AFTER: scale only valid tiles.
M={M}, N={N}, TILE_P={TILE_P}, VALID_TILES={VALID_TILES}
total_tiles={TOTAL_TILES} but loop bounded to VALID_TILES.
"""
import nki
import nki.isa as nisa
import nki.language as nl

M = {M}
N = {N}
TILE_P = {TILE_P}
TOTAL_TILES = {TOTAL_TILES}
VALID_TILES = {VALID_TILES}


@nki.jit(platform_target="trn1")
def scale_valid_tiles(x):
    y = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    for i in nl.sequential_range(VALID_TILES):
        x_tile = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tile, src=x[i * TILE_P:(i + 1) * TILE_P, 0:N])

        result = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=result, data=x_tile, op0=nl.multiply, operand0=2.0)

        nisa.dma_copy(dst=y[i * TILE_P:(i + 1) * TILE_P, 0:N], src=result)

    return y
'''


def main() -> None:
    here = pathlib.Path(__file__).parent
    for M, N, TILE_P, VALID_TILES in VARIANTS:
        if M % TILE_P != 0:
            raise SystemExit(f"M={M} must be divisible by TILE_P={TILE_P}")
        TOTAL_TILES = M // TILE_P
        if VALID_TILES <= 0 or VALID_TILES >= TOTAL_TILES:
            raise SystemExit(
                f"VALID_TILES={VALID_TILES} must satisfy 0 < VALID_TILES < TOTAL_TILES={TOTAL_TILES}"
            )
        ctx = dict(M=M, N=N, TILE_P=TILE_P, TOTAL_TILES=TOTAL_TILES, VALID_TILES=VALID_TILES)
        base = f"scale_M{M}_N{N}_T{TILE_P}_V{VALID_TILES}"
        (here / f"{base}_before.py").write_text(_BEFORE_TEMPLATE.format(**ctx))
        (here / f"{base}_after.py").write_text(_AFTER_TEMPLATE.format(**ctx))
        print(f"  wrote {base}_before.py + {base}_after.py")


if __name__ == "__main__":
    main()
