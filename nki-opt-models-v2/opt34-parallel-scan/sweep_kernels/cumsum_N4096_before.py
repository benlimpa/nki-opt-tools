"""
Sweep variant — opt34 BEFORE: sequential row-by-row 1D cumsum.
N_TOTAL=4096, P=128, TILE_N=32.
"""
import nki
import nki.isa as nisa
import nki.language as nl

N_TOTAL = 4096
P = 128
TILE_N = 32


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
