"""
Sweep variant — opt34 AFTER: parallel 1D cumsum via [P, TILE_N] reshape +
nc_matmul carry-over trick.
N_TOTAL=16384, P=128, TILE_N=128.
"""
import nki
import nki.isa as nisa
import nki.language as nl

N_TOTAL = 16384
P = 128
TILE_N = 128


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
