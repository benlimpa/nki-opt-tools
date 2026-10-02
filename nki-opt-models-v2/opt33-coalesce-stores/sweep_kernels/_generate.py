#!/usr/bin/env python3
"""
Generate parameterized coalesce-stores before/after kernels.

Pattern: y[m, frag*FRAG_W : (frag+1)*FRAG_W] = x[same range] * 2.0
- Before: each frag stored separately (n_fragments per iteration).
- After: all frags coalesced into a wide SBUF tile, stored once.

VARIANTS list: (n_iters, n_fragments_per_iter) tuples.
TILE_M=128 fixed; FRAG_W=128 fixed; row width = n_fragments * FRAG_W.
"""

from __future__ import annotations

import pathlib

TILE_M = 128
FRAG_W = 128

VARIANTS = [
    # (n_iters, n_fragments_per_iter)
    (64, 2),   # v1 anchor
    (32, 2),   # fewer iters
    (128, 2),  # more iters
    (64, 4),   # more fragments per iter
    (32, 4),   # both varied
]


def _stores_block(n_frag: int, indent: str) -> str:
    """Generate the per-iter stores section for the BEFORE kernel."""
    lines = []
    for f in range(n_frag):
        lines.append(
            f"{indent}r{f} = nl.ndarray((TILE_M, FRAG_W), dtype=nl.float32, buffer=nl.sbuf)"
        )
        lines.append(
            f"{indent}nisa.tensor_scalar(dst=r{f}, data=t{f}, op0=nl.multiply, operand0=2.0)"
        )
    for f in range(n_frag):
        lines.append(
            f"{indent}nisa.dma_copy(dst=out_ptr[0:TILE_M, "
            f"offset + {f}*FRAG_W : offset + {f+1}*FRAG_W], src=r{f})"
        )
    return "\n".join(lines)


def _loads_block(n_frag: int, indent: str) -> str:
    lines = []
    for f in range(n_frag):
        lines.append(
            f"{indent}t{f} = nl.ndarray((TILE_M, FRAG_W), dtype=nl.float32, buffer=nl.sbuf)"
        )
        lines.append(
            f"{indent}nisa.dma_copy(dst=t{f}, "
            f"src=in_ptr[0:TILE_M, offset + {f}*FRAG_W : offset + {f+1}*FRAG_W])"
        )
    return "\n".join(lines)


def _coalesce_block(n_frag: int, indent: str) -> str:
    lines = []
    for f in range(n_frag):
        lines.append(
            f"{indent}r{f} = nl.ndarray((TILE_M, FRAG_W), dtype=nl.float32, buffer=nl.sbuf)"
        )
        lines.append(
            f"{indent}nisa.tensor_scalar(dst=r{f}, data=t{f}, op0=nl.multiply, operand0=2.0)"
        )
    lines.append(
        f"{indent}combined = nl.ndarray((TILE_M, {n_frag}*FRAG_W), "
        f"dtype=nl.float32, buffer=nl.sbuf)"
    )
    for f in range(n_frag):
        lines.append(
            f"{indent}nisa.dma_copy(dst=combined[:, {f}*FRAG_W:{f+1}*FRAG_W], src=r{f})"
        )
    lines.append(
        f"{indent}nisa.dma_copy(dst=out_ptr[0:TILE_M, "
        f"offset:offset + {n_frag}*FRAG_W], src=combined)"
    )
    return "\n".join(lines)


_BEFORE_HEADER = '''\
"""
Sweep variant — opt33 BEFORE: {n_frag} fragmented stores per iter.
n_iters={n_iters}, n_fragments_per_iter={n_frag}.
"""
import nki
import nki.isa as nisa
import nki.language as nl

N_ITERS = {n_iters}
N_FRAG = {n_frag}
TILE_M = {TILE_M}
FRAG_W = {FRAG_W}
ROW_W = N_FRAG * FRAG_W
TOTAL_W = N_ITERS * ROW_W


@nki.jit(platform_target="trn1")
def coalesce_before(in_ptr):
    out_ptr = nl.ndarray((TILE_M, TOTAL_W), dtype=nl.float32, buffer=nl.shared_hbm)
    for i in nl.affine_range(N_ITERS):
        offset = i * ROW_W
{loads}
{stores}
    return out_ptr
'''

_AFTER_HEADER = '''\
"""
Sweep variant — opt33 AFTER: stores coalesced into single wide dma_copy.
n_iters={n_iters}, n_fragments_per_iter={n_frag}.
"""
import nki
import nki.isa as nisa
import nki.language as nl

N_ITERS = {n_iters}
N_FRAG = {n_frag}
TILE_M = {TILE_M}
FRAG_W = {FRAG_W}
ROW_W = N_FRAG * FRAG_W
TOTAL_W = N_ITERS * ROW_W


@nki.jit(platform_target="trn1")
def coalesce_after(in_ptr):
    out_ptr = nl.ndarray((TILE_M, TOTAL_W), dtype=nl.float32, buffer=nl.shared_hbm)
    for i in nl.affine_range(N_ITERS):
        offset = i * ROW_W
{loads}
{coalesce}
    return out_ptr
'''


def main() -> None:
    here = pathlib.Path(__file__).parent
    for n_iters, n_frag in VARIANTS:
        loads = _loads_block(n_frag, indent="        ")
        stores = _stores_block(n_frag, indent="        ")
        coalesce = _coalesce_block(n_frag, indent="        ")
        ctx = dict(n_iters=n_iters, n_frag=n_frag, TILE_M=TILE_M, FRAG_W=FRAG_W)
        before = _BEFORE_HEADER.format(loads=loads, stores=stores, **ctx)
        after = _AFTER_HEADER.format(loads=loads, coalesce=coalesce, **ctx)
        base = f"cs_iters{n_iters}_frag{n_frag}"
        (here / f"{base}_before.py").write_text(before)
        (here / f"{base}_after.py").write_text(after)
        print(f"  wrote {base}_before.py + {base}_after.py")


if __name__ == "__main__":
    main()
