import nki as nki
import nki.isa as nisa
import nki.language as nl
import numpy as np
import torch
from torch_xla.core import xla_model as xm


@nki.jit(platform_target="trn1")
def layer_norm_nki(X_hbm, W_hbm, B_hbm, Mean_shape, Rstd_shape, stride, N, eps, BLOCK_SIZE):
    M = X_hbm.shape[0]

    Y_hbm = nl.ndarray(X_hbm.shape, dtype=X_hbm.dtype, buffer=nl.shared_hbm)
    Mean_hbm = nl.ndarray(Mean_shape, dtype=nl.float32, buffer=nl.shared_hbm)
    Rstd_hbm = nl.ndarray(Rstd_shape, dtype=nl.float32, buffer=nl.shared_hbm)

    TILE_M = 128
    invN = 1.0 / float(N)

    for m in nl.affine_range((M + TILE_M - 1) // TILE_M):
        row_start = m * TILE_M
        row_end = min(M, row_start + TILE_M)
        rows = row_end - row_start

        x_tile = nl.ndarray((rows, N), dtype=nl.float32, buffer=nl.sbuf)
        w_tile = nl.ndarray((rows, N), dtype=nl.float32, buffer=nl.sbuf)
        b_tile = nl.ndarray((rows, N), dtype=nl.float32, buffer=nl.sbuf)

        sum_tile = nl.ndarray((rows, 1), dtype=nl.float32, buffer=nl.sbuf)
        mean_tile = nl.ndarray((rows, 1), dtype=nl.float32, buffer=nl.sbuf)
        var_sum_tile = nl.ndarray((rows, 1), dtype=nl.float32, buffer=nl.sbuf)
        var_tile = nl.ndarray((rows, 1), dtype=nl.float32, buffer=nl.sbuf)
        rstd_tile = nl.ndarray((rows, 1), dtype=nl.float32, buffer=nl.sbuf)

        centered = nl.ndarray((rows, N), dtype=nl.float32, buffer=nl.sbuf)
        sq_tile = nl.ndarray((rows, N), dtype=nl.float32, buffer=nl.sbuf)
        norm_tile = nl.ndarray((rows, N), dtype=nl.float32, buffer=nl.sbuf)
        y_tile = nl.ndarray((rows, N), dtype=nl.float32, buffer=nl.sbuf)

        nisa.dma_copy(dst=x_tile, src=X_hbm[row_start:row_end, 0:N])
        nisa.dma_copy(dst=w_tile, src=W_hbm[row_start:row_end, 0:N])
        nisa.dma_copy(dst=b_tile, src=B_hbm[row_start:row_end, 0:N])

        nisa.tensor_reduce(dst=sum_tile, op=nl.add, data=x_tile, axis=1)
        nisa.tensor_scalar(dst=mean_tile, data=sum_tile, op0=nl.multiply, operand0=invN)

        nisa.tensor_scalar(dst=centered, data=x_tile, op0=nl.subtract, operand0=mean_tile)
        nisa.tensor_tensor(dst=sq_tile, data1=centered, data2=centered, op=nl.multiply)
        nisa.tensor_reduce(dst=var_sum_tile, op=nl.add, data=sq_tile, axis=1)
        nisa.tensor_scalar(dst=var_tile, data=var_sum_tile, op0=nl.multiply, operand0=invN)
        nisa.tensor_scalar(dst=var_tile, data=var_tile, op0=nl.add, operand0=eps)
        nisa.activation(dst=rstd_tile, op=nl.rsqrt, data=var_tile)

        nisa.tensor_scalar(dst=norm_tile, data=centered, op0=nl.multiply, operand0=rstd_tile)
        nisa.tensor_tensor(dst=y_tile, data1=norm_tile, data2=w_tile, op=nl.multiply)
        nisa.tensor_tensor(dst=y_tile, data1=y_tile, data2=b_tile, op=nl.add)

        nisa.dma_copy(dst=Y_hbm[row_start:row_end, 0:N], src=y_tile)
        nisa.dma_copy(dst=Mean_hbm[row_start:row_end], src=mean_tile)
        nisa.dma_copy(dst=Rstd_hbm[row_start:row_end], src=rstd_tile)

    return Y_hbm, Mean_hbm, Rstd_hbm


# Wrapper Function:
def layer_norm(x, weight, bias, eps):
    device = xm.xla_device()
    x = x.to(device)
    weight = weight.to(device)
    bias = bias.to(device)

    x_arg = x.reshape(-1, x.shape[-1]).contiguous()
    M, N = x_arg.shape

    # Expand weight and bias to full row-wise shapes to avoid broadcast limitations.
    weight_arg = weight.reshape(1, -1).expand(M, N).contiguous()
    bias_arg = bias.reshape(1, -1).expand(M, N).contiguous()

    y, mean, rstd = layer_norm_nki[(1,)](
        X_hbm=x_arg,
        W_hbm=weight_arg,
        B_hbm=bias_arg,
        Mean_shape=(M,),
        Rstd_shape=(M,),
        stride=x_arg.stride(0),
        N=N,
        eps=eps,
        BLOCK_SIZE=128,
    )
    return y.reshape(x.shape)


if __name__ == "__main__":
    x = torch.randn((512, 2048), dtype=torch.float32)
    w = torch.ones((2048,), dtype=torch.float32)
    b = torch.zeros((2048,), dtype=torch.float32)
    result = layer_norm(x, w, b, 1e-5)
    xm.mark_step()
    print(f"before.py output shape={result.shape}  dtype={result.dtype}")
