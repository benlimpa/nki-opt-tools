"""
Shared oracle for opt13 sweep variants.

Reads M, K, N from kernel module globals. Tests C = lhsT.T @ rhs where
lhsT is [K, M] (transposed A) and rhs is [K, N]. The before kernel
expects f32 inputs; the after kernel expects bf16 inputs. The oracle
casts inputs to bf16 when the kernel name contains 'bf16'.
"""
import torch
import torch_xla.core.xla_model as xm


def _make_inputs(device, kernel_dtype):
    M_, K_, N_ = int(M), int(K), int(N)  # noqa: F821 — read from kernel module
    torch.manual_seed(0)
    A = torch.randn(M_, K_, dtype=torch.float32)
    B = torch.randn(K_, N_, dtype=torch.float32)
    lhsT = A.T.contiguous()
    if kernel_dtype is torch.bfloat16:
        lhsT_dev = lhsT.to(torch.bfloat16).to(device)
        rhs_dev = B.to(torch.bfloat16).to(device)
    else:
        lhsT_dev = lhsT.to(device)
        rhs_dev = B.to(device)
    return A, B, lhsT_dev, rhs_dev


def _kernel_dtype(wrapper):
    # Heuristic: function name carries the dtype tag in this sweep.
    name = getattr(wrapper, "__name__", "") or ""
    if "bf16" in name or "bfloat16" in name:
        return torch.bfloat16
    return torch.float32


def run_test(wrapper):
    device = xm.xla_device()
    kdt = _kernel_dtype(wrapper)
    A, B, lhsT_dev, rhs_dev = _make_inputs(device, kdt)
    C = wrapper(lhsT_dev, rhs_dev)
    xm.mark_step()
    C_cpu = C.cpu()
    ref = A @ B
    # bf16 inputs degrade accuracy noticeably — relax tolerance.
    atol = 1e-2 if kdt is torch.float32 else 5.0
    rtol = 1e-3 if kdt is torch.float32 else 1e-1
    torch.testing.assert_close(C_cpu, ref, atol=atol, rtol=rtol)
    print("Test passed")


def bench(wrapper):
    device = xm.xla_device()
    kdt = _kernel_dtype(wrapper)
    _, _, lhsT_dev, rhs_dev = _make_inputs(device, kdt)
    C = wrapper(lhsT_dev, rhs_dev)
    xm.mark_step()
    _ = C.cpu()
    print("Bench setup passed")
