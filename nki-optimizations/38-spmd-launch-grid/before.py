"""
BEFORE kernel: Elementwise add launched on a single NeuronCore.

Optimization target: #38 — SPMD launch grid.

This kernel adds two [M, N] tensors row-wise, tiling M into TILE_P=128 chunks.
The BEFORE version is launched WITHOUT an SPMD grid, so a single physical
NeuronCore executes every iteration of the outer loop sequentially.

The outer loop iterations are independent — each writes a disjoint TILE_P-row
slice of the output — so they are trivially shardable across NCs.
"""

import nki
import nki.isa as nisa
import nki.language as nl

M = 2048     # total rows (must be multiple of TILE_P and of every candidate grid size)
N = 512
TILE_P = 128


@nki.jit(platform_target="trn1")
def add_no_spmd(x, y):
    """
    Compute z[i, j] = x[i, j] + y[i, j] for [M, N] inputs.

    BEFORE: no SPMD launch grid. One NeuronCore handles all M // TILE_P iterations.

    x, y: [M, N] in HBM
    Returns: z [M, N] in HBM
    """
    z = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    n_iters = M // TILE_P
    for i in nl.sequential_range(n_iters):
        x_tile = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)
        y_tile = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tile, src=x[i * TILE_P:(i + 1) * TILE_P, 0:N])
        nisa.dma_copy(dst=y_tile, src=y[i * TILE_P:(i + 1) * TILE_P, 0:N])

        z_tile = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=z_tile, data1=x_tile, data2=y_tile, op=nl.add)

        nisa.dma_copy(dst=z[i * TILE_P:(i + 1) * TILE_P, 0:N], src=z_tile)

    return z


def wrapper(x, y):
    return add_no_spmd(x, y)
