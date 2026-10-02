#!/usr/bin/env python3
"""
Generate parameterized loop-fusion before/after kernels for opt07.

Usage:  python _generate.py

VARIANTS list: (M, N, TILE_P, N_INTERMEDIATES) tuples. The before kernel
runs N_INTERMEDIATES + 1 separate sequential loops with HBM intermediates
between them. The after kernel fuses them into one loop.
"""

from __future__ import annotations

import pathlib

VARIANTS = [
    # (M, N, TILE_P, N_INTERMEDIATES)
    (1024,  512, 128, 1),   # 2-loop fusion
    (1024,  512, 128, 2),   # 3-loop fusion (v1 anchor)
    (1024,  512, 128, 3),   # 4-loop fusion
    (2048,  512, 128, 2),   # off-axis: larger M
    (1024, 1024, 128, 2),   # off-axis: larger N
]


_HEADER_TEMPLATE = '''\
"""
Sweep variant — opt07 {SUFFIX}: {N_LOOPS}-loop pipeline,
M={M}, N={N}, TILE_P={TILE_P}, n_intermediates={N_INTERMEDIATES}.
"""
import nki
import nki.isa as nisa
import nki.language as nl

M = {M}
N = {N}
TILE_P = {TILE_P}
N_OPS = {N_LOOPS}
'''


def _scalar_op_for(i: int) -> tuple[str, float]:
    """Return (nl op name, operand) for op index i. Mirrors pipe_oracle._ref_output."""
    if i % 2 == 0:
        return "nl.multiply", 2.0 - 0.5 * i
    else:
        return "nl.add", 1.0 - 0.5 * (i - 1)


def _gen_before(M: int, N: int, TILE_P: int, n_int: int) -> str:
    n_loops = n_int + 1
    header = _HEADER_TEMPLATE.format(
        SUFFIX="BEFORE", N_LOOPS=n_loops, M=M, N=N, TILE_P=TILE_P,
        N_INTERMEDIATES=n_int,
    )
    body = ["", "@nki.jit(platform_target=\"trn1\")",
            "def pipeline_separate(x):",
            "    y = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)"]
    # Allocate n_int HBM intermediates
    for k in range(n_int):
        body.append(f"    inter{k} = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.private_hbm)")
    body.append("")
    # Each pass uses src_buf (HBM) and dst_buf (HBM)
    src_bufs = ["x"] + [f"inter{k}" for k in range(n_int)]
    dst_bufs = [f"inter{k}" for k in range(n_int)] + ["y"]
    for pass_i in range(n_loops):
        op_name, operand = _scalar_op_for(pass_i)
        src = src_bufs[pass_i]
        dst = dst_bufs[pass_i]
        body.append(f"    # Pass {pass_i}: {op_name} {operand}")
        body.append("    for i in nl.affine_range(M // TILE_P):")
        body.append("        t_in = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)")
        body.append(f"        nisa.dma_copy(dst=t_in, src={src}[i * TILE_P:(i + 1) * TILE_P, 0:N])")
        body.append("        t_out = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)")
        body.append(f"        nisa.tensor_scalar(dst=t_out, data=t_in, op0={op_name}, operand0={operand!r})")
        body.append(f"        nisa.dma_copy(dst={dst}[i * TILE_P:(i + 1) * TILE_P, 0:N], src=t_out)")
        body.append("")
    body.append("    return y")
    body.append("")
    return header + "\n".join(body)


def _gen_after(M: int, N: int, TILE_P: int, n_int: int) -> str:
    n_loops = n_int + 1
    header = _HEADER_TEMPLATE.format(
        SUFFIX="AFTER", N_LOOPS=n_loops, M=M, N=N, TILE_P=TILE_P,
        N_INTERMEDIATES=n_int,
    )
    body = ["", "@nki.jit(platform_target=\"trn1\")",
            "def pipeline_fused(x):",
            "    y = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)",
            "",
            "    for i in nl.affine_range(M // TILE_P):",
            "        x_tile = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)",
            "        nisa.dma_copy(dst=x_tile, src=x[i * TILE_P:(i + 1) * TILE_P, 0:N])",
            "        cur = x_tile"]
    for pass_i in range(n_loops):
        op_name, operand = _scalar_op_for(pass_i)
        body.append(f"        # Op {pass_i}: {op_name} {operand}")
        body.append(f"        nxt_{pass_i} = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)")
        body.append(f"        nisa.tensor_scalar(dst=nxt_{pass_i}, data=cur, op0={op_name}, operand0={operand!r})")
        body.append(f"        cur = nxt_{pass_i}")
    body.append("        nisa.dma_copy(dst=y[i * TILE_P:(i + 1) * TILE_P, 0:N], src=cur)")
    body.append("")
    body.append("    return y")
    body.append("")
    return header + "\n".join(body)


def main() -> None:
    here = pathlib.Path(__file__).parent
    for M, N, TILE_P, n_int in VARIANTS:
        if M % TILE_P != 0:
            raise SystemExit(f"M={M} must be divisible by TILE_P={TILE_P}")
        base = f"pipe_M{M}_N{N}_int{n_int}"
        (here / f"{base}_before.py").write_text(_gen_before(M, N, TILE_P, n_int))
        (here / f"{base}_after.py").write_text(_gen_after(M, N, TILE_P, n_int))
        print(f"  wrote {base}_before.py + {base}_after.py")


if __name__ == "__main__":
    main()
