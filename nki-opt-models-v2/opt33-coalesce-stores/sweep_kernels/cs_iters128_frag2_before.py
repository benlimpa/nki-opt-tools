"""
Sweep variant — opt33 BEFORE: 2 fragmented stores per iter.
n_iters=128, n_fragments_per_iter=2.
"""
import nki
import nki.isa as nisa
import nki.language as nl

N_ITERS = 128
N_FRAG = 2
TILE_M = 128
FRAG_W = 128
ROW_W = N_FRAG * FRAG_W
TOTAL_W = N_ITERS * ROW_W


@nki.jit(platform_target="trn1")
def coalesce_before(in_ptr):
    out_ptr = nl.ndarray((TILE_M, TOTAL_W), dtype=nl.float32, buffer=nl.shared_hbm)
    for i in nl.affine_range(N_ITERS):
        offset = i * ROW_W
        t0 = nl.ndarray((TILE_M, FRAG_W), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=t0, src=in_ptr[0:TILE_M, offset + 0*FRAG_W : offset + 1*FRAG_W])
        t1 = nl.ndarray((TILE_M, FRAG_W), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=t1, src=in_ptr[0:TILE_M, offset + 1*FRAG_W : offset + 2*FRAG_W])
        r0 = nl.ndarray((TILE_M, FRAG_W), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=r0, data=t0, op0=nl.multiply, operand0=2.0)
        r1 = nl.ndarray((TILE_M, FRAG_W), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=r1, data=t1, op0=nl.multiply, operand0=2.0)
        nisa.dma_copy(dst=out_ptr[0:TILE_M, offset + 0*FRAG_W : offset + 1*FRAG_W], src=r0)
        nisa.dma_copy(dst=out_ptr[0:TILE_M, offset + 1*FRAG_W : offset + 2*FRAG_W], src=r1)
    return out_ptr
