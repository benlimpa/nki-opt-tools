"""
Shared oracle for opt03 sweep variants.

Reads M, N from the kernel module's globals. Tests element-wise A + B
where A, B are [M, N] float32.
"""
import torch
import torch_xla.core.xla_model as xm


def _make_inputs(device):
    M_, N_ = int(M), int(N)  # noqa: F821 — read from kernel module
    torch.manual_seed(0)
    A = torch.randn(M_, N_, dtype=torch.float32)
    B = torch.randn(M_, N_, dtype=torch.float32)
    return A, B, A.to(device), B.to(device)


def run_test(wrapper):
    device = xm.xla_device()
    A, B, A_dev, B_dev = _make_inputs(device)
    C = wrapper(A_dev, B_dev)
    xm.mark_step()
    C_cpu = C.cpu()
    ref = A + B
    torch.testing.assert_close(C_cpu, ref, atol=1e-5, rtol=1e-5)
    print("Test passed")


def bench(wrapper):
    device = xm.xla_device()
    _, _, A_dev, B_dev = _make_inputs(device)
    C = wrapper(A_dev, B_dev)
    xm.mark_step()
    _ = C.cpu()  # force XLA materialization → NEFF emission
    print("Bench setup passed")
