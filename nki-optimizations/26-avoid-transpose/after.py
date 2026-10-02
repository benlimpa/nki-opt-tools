"""
AFTER kernel: Keep data in identical memory layout to avoid runtime transposes.

The kernel expects the LHS input strictly natively preserved in the [K, M] layout.
Because the hardware stationary memory partition exactly aligns with K, the kernel
can invoke `nisa.nc_matmul` directly against the raw fetched tile without executing 
any intermediate permutation or rotation. 

This saves 50% of the Tensor Engine cycles inside the loop.
"""

import nki
import nki.language as nl
import nki.isa as nisa

M = 2048
K = 128
N = 512
P_DIM = 128

@nki.jit(platform_target="trn1")
def after_kernel(x_T, y):
    # x_T: [K, M] -> [128, 16384]
    # y: [K, N] -> [128, 512]
    res = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)
    
    y_tile = nl.ndarray((K, N), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=y_tile, src=y)
    
    for m in nl.affine_range(M // P_DIM):
        # x_T_tile is natively [K, M_chunk] -> [128, 128]
        x_tile = nl.ndarray((P_DIM, P_DIM), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tile, src=x_T[0:K, m*P_DIM:(m+1)*P_DIM])
        
        # Invoke low-level matrix-multiplication directly. NO runtime transpose necessary!
        res_psum = nl.ndarray((P_DIM, N), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=res_psum, stationary=x_tile, moving=y_tile)

        res_tile = nl.ndarray((P_DIM, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=res_tile, data=res_psum, op=nl.copy)
        
        nisa.dma_copy(dst=res[m*P_DIM:(m+1)*P_DIM, :], src=res_tile)
        
    return res

if __name__ == "__main__":
    import torch
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()
    torch.manual_seed(0)

    # Note: User pre-transposes the matrix or natively stores it as [K, M]
    x = torch.randn(M, K, dtype=torch.float32)
    x_T = x.transpose(0, 1).contiguous().to(device)
    y = torch.randn(K, N, dtype=torch.float32).to(device)

    res_dev = after_kernel(x_T, y)
    xm.mark_step()
    
    ref_res = torch.matmul(x, y.cpu())
    out_res = res_dev.cpu()
    
    print("Max diff:", torch.max(torch.abs(out_res - ref_res)))
    assert torch.allclose(out_res, ref_res, atol=1e-2, rtol=1e-2), "Mismatch"
    print("Correctness check passed.")
