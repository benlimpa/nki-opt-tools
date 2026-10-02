import nki as nki
import nki.isa as nisa
import nki.language as nl
import numpy as np
import torch
import torch_xla
from torch_xla.core import xla_model as xm


@nki.jit(platform_target="trn1")
def _rms_layernorm_forward_nki(X_hbm, X_row_stride, W_hbm, W_row_stride, n_cols, eps, BLOCK_SIZE):
    """
    Computes RMS Layer Normalization.

    Returns:
        Y_hbm: Normalized output tensor of shape (rows, n_cols)
        r_hbm: Reciprocal RMS for each row
    """
    n_rows = X_hbm.shape[0]

    Y_hbm = nl.ndarray(X_hbm.shape, dtype=X_hbm.dtype, buffer=nl.shared_hbm)
    r_hbm = nl.ndarray((n_rows, 1), dtype=nl.float32, buffer=nl.shared_hbm)

    # Assume W_hbm is passed as shape (1, n_cols) from the wrapper.
    w_tile = nl.ndarray((1, n_cols), dtype=W_hbm.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=w_tile, src=W_hbm[0:1, 0:n_cols])

    for r in nl.affine_range(n_rows):
        # Load one row of X
        x_row = nl.ndarray((1, n_cols), dtype=X_hbm.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_row, src=X_hbm[r:r + 1, 0:n_cols])

        # x^2
        x_sq = nl.ndarray((1, n_cols), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=x_sq, data1=x_row, data2=x_row, op=nl.multiply)

        # sum(x^2) over free dimension
        sum_sq = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=sum_sq, op=nl.add, data=x_sq, axis=1, keepdims=True)

        # mean(x^2) + eps
        mean_sq = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=mean_sq, data=sum_sq, op0=nl.multiply, operand0=(1.0 / float(n_cols)))

        mean_sq_eps = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=mean_sq_eps, data=mean_sq, op0=nl.add, operand0=eps)

        # inv_rms = rsqrt(mean(x^2) + eps)
        inv_rms = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=inv_rms, op=nl.rsqrt, data=mean_sq_eps)

        # Store reciprocal RMS
        nisa.dma_copy(dst=r_hbm[r:r + 1, 0:1], src=inv_rms)

        # x_norm = x * inv_rms
        x_norm = nl.ndarray((1, n_cols), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=x_norm, data=x_row, op0=nl.multiply, operand0=inv_rms)

        # y_row = x_norm * w_tile
        y_row = nl.ndarray((1, n_cols), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=y_row, data1=x_norm, data2=w_tile, op=nl.multiply)

        # Store output
        nisa.dma_copy(dst=Y_hbm[r:r + 1, 0:n_cols], src=y_row)

    return Y_hbm, r_hbm


# Wrapper Function:
def rms_layernorm(X, W, eps=1e-5):
    device = xm.xla_device()
    X = X.to(device)
    W = W.to(device)

    assert X.dim() == 2, "X must be 2D"
    assert W.dim() == 1, "W must be 1D"
    n_rows, n_cols = X.shape
    assert W.shape[0] == n_cols, "W must have the same number of columns as X"

    # Pass W as a 2D row vector to match the kernel's expected DMA shape.
    W2 = W.unsqueeze(0)

    y, r = _rms_layernorm_forward_nki(
        X_hbm=X,
        X_row_stride=X.stride(0),
        W_hbm=W2,
        W_row_stride=W2.stride(0),
        n_cols=n_cols,
        eps=eps,
        BLOCK_SIZE=n_cols,
    )
    return y


if __name__ == "__main__":
    # Benchmark input: (512, 2048) — matches rms_logic.py bench()
    X = torch.randn(512, 2048)
    W = torch.ones(2048)
    result = rms_layernorm(X, W, eps=1e-5)
    torch_xla.sync()
    print(f"before.py output shape={result.shape}  dtype={result.dtype}")
