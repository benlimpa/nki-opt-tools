"""
BEFORE kernel: Explicit array materialization.

The developer allocates explicitly isolated `nl.ndarray` targets for every
incremental mathematical step in the operation sequence. Mathematically, this 
mimics SSA (Static Single Assignment) form but explicitly demands fresh SBUF footprint allocations continuously.
"""

import nki
import nki.language as nl
import nki.isa as nisa

P = 128
TILE_N = 512
N = 131072

@nki.jit(platform_target="trn1")
def inplace_before(x_in, y_in):
    res_out = nl.ndarray((P, N), dtype=nl.float32, buffer=nl.shared_hbm)
    
    for n in nl.sequential_range(N // TILE_N):
        x = nl.ndarray((P, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        y = nl.ndarray((P, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        
        nisa.dma_copy(dst=x, src=x_in[0:P, n*TILE_N:(n+1)*TILE_N])
        nisa.dma_copy(dst=y, src=y_in[0:P, n*TILE_N:(n+1)*TILE_N])
        
        # Explicit allocations
        z1 = nl.ndarray((P, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=z1, data1=x, data2=y, op=nl.multiply)
        
        z2 = nl.ndarray((P, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=z2, data1=z1, data2=x, op=nl.add)
        
        z3 = nl.ndarray((P, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=z3, data1=z2, data2=y, op=nl.subtract)
        
        nisa.dma_copy(dst=res_out[0:P, n*TILE_N:(n+1)*TILE_N], src=z3)

    return res_out

if __name__ == "__main__":
    import torch
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()
    torch.manual_seed(0)

    x = torch.randn(P, N, dtype=torch.float32).to(device)
    y = torch.randn(P, N, dtype=torch.float32).to(device)

    res_dev = inplace_before(x, y)
    xm.mark_step()
    
    ref_res = (x.cpu() * y.cpu()) + x.cpu() - y.cpu()
    out_res = res_dev.cpu()
    
    print("Max diff:", torch.max(torch.abs(out_res - ref_res)))
    assert torch.allclose(out_res, ref_res, atol=1e-3, rtol=1e-3), "Mismatch"
    print("Correctness check passed.")
