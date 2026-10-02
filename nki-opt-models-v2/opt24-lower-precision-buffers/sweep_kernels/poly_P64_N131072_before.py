"""
Sweep variant — opt24 BEFORE: fp32 internal SBUF tiles (implicit casts).
P=64, N=131072, TILE_N=512.
"""
import nki
import nki.isa as nisa
import nki.language as nl

P = 64
N = 131072
TILE_N = 512


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
