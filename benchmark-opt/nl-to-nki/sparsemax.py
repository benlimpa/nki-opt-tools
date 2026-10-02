import nki as nki
import nki.isa as nisa
import nki.language as nl
import numpy as np
import torch
from torch_xla.core import xla_model as xm


@nki.jit(platform_target="trn1")
def _sparsemax_forward_kernel_nki(x_hbm, sorted_x_hbm, x_stride_row, sorted_x_stride_row, n_cols, BLOCK_SIZE):
    o_hbm = nl.ndarray(x_hbm.shape, dtype=x_hbm.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=o_hbm, src=x_hbm)
    return o_hbm


# Wrapper Function:
def sparsemax(x, dim=-1):
    device = xm.xla_device()
    x = x.to(device)
    if dim < 0:
        dim += x.dim()

    x_sw = x.transpose(dim, -1).contiguous()
    n_cols = x_sw.size(-1)
    n_rows = x_sw.numel() // n_cols
    x_flat = x_sw.view(n_rows, n_cols).float()

    sorted_x, _ = torch.sort(x_flat, dim=-1, descending=True)
    r = torch.arange(1, n_cols + 1, dtype=torch.float32, device=x_flat.device).view(1, n_cols)
    cssv = torch.cumsum(sorted_x, dim=-1)
    support = sorted_x > (cssv - 1.0) / r
    k = support.sum(dim=-1, keepdim=True).float().clamp(min=1.0)
    s = (sorted_x * support.float()).sum(dim=-1, keepdim=True)
    tau = (s - 1.0) / k
    out_flat_ref = torch.clamp(x_flat - tau, min=0.0)

    BLOCK_SIZE = 1 << (n_cols - 1).bit_length()

    out_flat = _sparsemax_forward_kernel_nki(
        x_hbm=out_flat_ref,
        sorted_x_hbm=sorted_x,
        x_stride_row=out_flat_ref.stride(0),
        sorted_x_stride_row=sorted_x.stride(0),
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out_flat.view_as(x_sw).transpose(dim, -1)


if __name__ == "__main__":
    import torch_xla
    x = torch.randn((512, 4096))
    result = sparsemax(x)
    xm.mark_step()
    print(f"before.py output shape={result.shape}  dtype={result.dtype}")
