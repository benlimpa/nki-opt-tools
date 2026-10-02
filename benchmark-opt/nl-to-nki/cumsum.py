import nki as nki
import nki.isa as nisa
import nki.language as nl
import numpy as np
import torch
from torch_xla.core import xla_model as xm


@nki.jit(platform_target="trn1")
def chunk_local_cumsum_scalar_kernel_nki(s_row_hbm, T, BT):
    out_row_hbm = nl.ndarray(s_row_hbm.shape, dtype=s_row_hbm.dtype, buffer=nl.shared_hbm)

    num_chunks = (T + BT - 1) // BT
    running = 0.0

    for ck in nl.sequential_range(num_chunks):
        start = ck * BT
        end = start + BT
        valid = BT
        if end > T:
            valid = T - start

        x_tile = nl.ndarray((1, BT), dtype=s_row_hbm.dtype, buffer=nl.sbuf)
        y_tile = nl.ndarray((1, BT), dtype=s_row_hbm.dtype, buffer=nl.sbuf)

        nisa.memset(x_tile, 0)
        nisa.memset(y_tile, 0)

        nisa.dma_copy(dst=x_tile[:, 0:valid], src=s_row_hbm[:, start:end])

        # Compute local prefix sum for this chunk, then add the running offset.
        nisa.tensor_scalar_cumulative(
            dst=y_tile,
            src=x_tile,
            op0=nl.add,
            op1=nl.add,
            imm0=0.0,
            reduce_cmd=nisa.reduce_cmd.reset_reduce,
        )

        if running != 0.0:
            nisa.tensor_scalar(
                dst=y_tile,
                data=y_tile,
                op0=nl.add,
                operand0=running,
            )

        nisa.dma_copy(dst=out_row_hbm[:, start:end], src=y_tile[:, 0:valid])

        # Update running prefix sum using the last valid element of this chunk.
        # The final element of the cumulative tile equals the total chunk sum.
        total_tile = nl.ndarray((1, 1), dtype=s_row_hbm.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=total_tile, src=y_tile[:, valid - 1:valid])

        # Store into a Python scalar-like value through a single-element tensor copy
        # by reading from the one-element tile on the next iteration via `tensor_scalar`.
        # Since NKI doesn't support direct Python-side tensor reads reliably, we use
        # a conservative cumulative update trick with a single-element tile.
        if ck < num_chunks - 1:
            tmp_running = nl.ndarray((1, 1), dtype=s_row_hbm.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=tmp_running, src=total_tile)
            # Use the total value as the new running offset by adding it to itself
            # is unnecessary; instead keep the chunk total in a 1x1 tile and apply it
            # through a scalar_tensor operation in the next chunk via `running_tile`.
            running = 0.0

    return out_row_hbm


# Wrapper Function:
def cumsum(g, chunk_size):
    device = xm.xla_device()
    g = g.to(device)
    B, H, T = g.shape

    assert chunk_size == 2 ** (
        chunk_size.bit_length() - 1
    ), "chunk_size must be a power of 2"
    BT = chunk_size

    rows = []
    g_flat = g.reshape(B * H, T)
    for i in range(B * H):
        row = g_flat[i:i + 1, :]
        out_row = chunk_local_cumsum_scalar_kernel_nki(row, T=T, BT=BT)
        rows.append(out_row)

    out_flat = torch.cat(rows, dim=0)
    return out_flat.reshape(B, H, T)


if __name__ == "__main__":
    import torch_xla
    # (4, 8, 2048) with chunk_size=64: bench input.
    # before.py makes 32 NKI kernel calls (one per row), each with shape (1, 2048).
    # The profiler captures one such call — a single-partition kernel with 32 chunks.
    g = torch.randn(4, 8, 2048)
    result = cumsum(g, chunk_size=64)
    xm.mark_step()
    print(f"before.py output shape={result.shape}  dtype={result.dtype}")
