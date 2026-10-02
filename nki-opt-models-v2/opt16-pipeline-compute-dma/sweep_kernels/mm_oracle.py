"""
Shared oracle for opt16 sweep variants.

Reads M, K, N, DTYPE from the kernel module's globals (nki-profile concatenates
kernel and oracle into one Python script before exec).
"""
import torch
import torch_xla.core.xla_model as xm

_TORCH_DTYPE = {"fp32": torch.float32, "bf16": torch.bfloat16}


def _shapes():
    return int(M), int(K), int(N), _TORCH_DTYPE[DTYPE]  # noqa: F821


def _make_inputs(device):
    M_, K_, N_, dt = _shapes()
    torch.manual_seed(0)
    A = torch.randn(M_, K_, dtype=torch.float32)
    B = torch.randn(K_, N_, dtype=torch.float32)
    lhsT = A.T.contiguous().to(dt).to(device)
    rhs = B.to(dt).to(device)
    return A, B, lhsT, rhs


def run_test(wrapper):
    device = xm.xla_device()
    A, B, lhsT, rhs = _make_inputs(device)
    C = wrapper(lhsT, rhs)
    xm.mark_step()
    ref = (A @ B)
    C_cpu = C.cpu().float()
    # bf16 needs a looser tolerance
    atol = 1e-3 if DTYPE == "fp32" else 5e-1  # noqa: F821
    rtol = 1e-3 if DTYPE == "fp32" else 5e-2  # noqa: F821
    torch.testing.assert_close(C_cpu, ref, atol=atol, rtol=rtol)
    print("Test passed")


def bench(wrapper):
    device = xm.xla_device()
    _, _, lhsT, rhs = _make_inputs(device)
    C = wrapper(lhsT, rhs)
    xm.mark_step()
    _ = C.cpu()  # force XLA materialization → NEFF emission
    print("Bench setup passed")
