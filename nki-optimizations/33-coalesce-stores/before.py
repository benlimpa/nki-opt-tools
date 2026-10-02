import torch
import nki
import nki.language as nl
import nki.isa as nisa
import numpy as np

@nki.jit(platform_target="trn1")
def coalesce_before(in_ptr):
    """
    Before: Store adjacent tiles separately
    """
    out_ptr = nl.ndarray((128, 64 * 256), dtype=nl.float32, buffer=nl.shared_hbm)
    for i in nl.affine_range(64):
        offset = i * 256

        # Load two adjacent tiles
        t0 = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=t0, src=in_ptr[0:128, offset : offset+128])
        t1 = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=t1, src=in_ptr[0:128, offset+128 : offset+256])

        # Some dummy compute
        r0 = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=r0, data=t0, op0=nl.multiply, operand0=2.0)

        r1 = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=r1, data=t1, op0=nl.multiply, operand0=2.0)

        # Store separately
        nisa.dma_copy(dst=out_ptr[0:128, offset : offset+128], src=r0)
        nisa.dma_copy(dst=out_ptr[0:128, offset+128 : offset+256], src=r1)

    return out_ptr

if __name__ == "__main__":
    import torch_xla.core.xla_model as xm
    device = xm.xla_device()
    import numpy as np
    M = 128
    N = 64 * 256

    in_t = torch.randn((M, N), dtype=torch.float32).to(device)

    out_t = coalesce_before(in_t)
    xm.mark_step()
    out_res = out_t.cpu()
    print("Before kernel completed, val:", out_res[0,0])
