"""
BEFORE kernel: Demanding runtime transpose.

The kernel receives its LHS input tensor in the logical layout [M, K].
However, the physical Tensor Engine requires the stationary argument's partition
dimension to represent the contraction axis K. Thus, the hardware requires an explicit
matrix transpose in SBUF before the actual matrix-multiplication can execute.
`nl.matmul` does this automatically when `transpose_x=True` is provided, but it 
silently consumes heavy Tensor Engine cycles to multiply by identity block matrices or routing.
"""

import nki
import nki.language as nl
import nki.isa as nisa

M = 2048
K = 128
N = 512
P_DIM = 128

@nki.jit(platform_target="trn1")
def before_kernel(x, y):
    # x: [M, K] -> [16384, 128]
    # y: [K, N] -> [128, 512]
    res = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)
    
    y_tile = nl.ndarray((K, N), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=y_tile, src=y)
    
    for m in nl.affine_range(M // P_DIM):
        x_tile = nl.ndarray((P_DIM, K), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tile, src=x[m*P_DIM:(m+1)*P_DIM, 0:K])
        
        x_transposed = nl.ndarray((K, P_DIM), dtype=nl.float32, buffer=nl.sbuf)
        nisa.nc_transpose(data=x_tile, dst=x_transposed, engine=nisa.tensor_engine)
        
        # High-level matmul API will dynamically insert a Tensor Engine transpose operation!
        res_psum = nl.ndarray((P_DIM, N), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=res_psum, stationary=x_transposed, moving=y_tile)
        
        res_tile = nl.ndarray((P_DIM, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=res_tile, data=res_psum, op=nl.copy)
        
        nisa.dma_copy(dst=res[m*P_DIM:(m+1)*P_DIM, :], src=res_tile)
        
    return res

if __name__ == "__main__":
    import torch
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()
    torch.manual_seed(0)

    x = torch.randn(M, K, dtype=torch.float32).to(device)
    y = torch.randn(K, N, dtype=torch.float32).to(device)

    res_dev = before_kernel(x, y)
    xm.mark_step()
    
    ref_res = torch.matmul(x.cpu(), y.cpu())
    out_res = res_dev.cpu()
    
    print("Max diff:", torch.max(torch.abs(out_res - ref_res)))
    assert torch.allclose(out_res, ref_res, atol=1e-2, rtol=1e-2), "Mismatch"
    print("Correctness check passed.")
