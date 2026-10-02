"""
AFTER kernel: Elementwise add launched on an SPMD grid across multiple NCs.

Optimization: #38 — SPMD launch grid.

KEY CHANGE: The wrapper invokes the kernel as `add_spmd[GRID_SIZE](x, y)`.
Inside the kernel, `nl.num_programs(0)` is the number of program instances
(one per logical NeuronCore) and `nl.program_id(0)` is the index of the
current instance. Each instance handles 1 / n_prgs of the outer-loop work.

The wrapper SWEEPS candidate grid sizes (typically `G ∈ {1, 2}`) and picks
the fastest — the right value depends on the available physical NeuronCores
on the current hardware (1 on Trainium1, 2 on Trainium2). On a single-NC
target, G=2 falls back to running both program instances on the one NC
serially and gains nothing — the sweep catches that.
"""

import nki
import nki.isa as nisa
import nki.language as nl

M = 2048     # total rows (must be divisible by TILE_P and by every candidate grid size)
N = 512
TILE_P = 128

# Sweep this on the target box; pick the value with the lowest measured latency.
GRID_SIZE = 2


@nki.jit(platform_target="trn1")
def add_spmd(x, y):
    """
    Compute z[i, j] = x[i, j] + y[i, j] for [M, N] inputs.

    AFTER: outer loop is sharded across `nl.num_programs(0)` program instances,
    one per logical NeuronCore.

    x, y: [M, N] in HBM
    Returns: z [M, N] in HBM
    """
    z = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    n_prgs = nl.num_programs(axes=0)
    prg_id = nl.program_id(axis=0)

    n_iters = M // TILE_P
    iters_per_prg = n_iters // n_prgs
    offset = prg_id * iters_per_prg

    for local_i in nl.affine_range(iters_per_prg):
        i = local_i + offset
        x_tile = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)
        y_tile = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tile, src=x[i * TILE_P:(i + 1) * TILE_P, 0:N])
        nisa.dma_copy(dst=y_tile, src=y[i * TILE_P:(i + 1) * TILE_P, 0:N])

        z_tile = nl.ndarray((TILE_P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=z_tile, data1=x_tile, data2=y_tile, op=nl.add)

        nisa.dma_copy(dst=z[i * TILE_P:(i + 1) * TILE_P, 0:N], src=z_tile)

    return z


def wrapper(x, y):
    return add_spmd[GRID_SIZE](x, y)
