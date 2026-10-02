"""
Shared oracle for opt14 sweep variants.

Reads M, K, N from kernel module globals. Tests C = lhsT.T @ rhs where
lhsT is [K, M] and rhs is [K, N]. K is small (single K-tile).
"""
import torch
import torch_xla.core.xla_model as xm


def _make_inputs(device):
    M_, K_, N_ = int(M), int(K), int(N)  # noqa: F821 — read from kernel module
    torch.manual_seed(0)
    A = torch.randn(M_, K_, dtype=torch.float32)
    B = torch.randn(K_, N_, dtype=torch.float32)
    lhsT = A.T.contiguous()
    return A, B, lhsT.to(device), B.to(device)


def run_test(wrapper):
    device = xm.xla_device()
    A, B, lhsT_dev, rhs_dev = _make_inputs(device)
    C = wrapper(lhsT_dev, rhs_dev)
    xm.mark_step()
    C_cpu = C.cpu()
    ref = A @ B
    torch.testing.assert_close(C_cpu, ref, atol=1e-2, rtol=1e-3)
    print("Test passed")


def bench(wrapper):
    device = xm.xla_device()
    _, _, lhsT_dev, rhs_dev = _make_inputs(device)
    C = wrapper(lhsT_dev, rhs_dev)
    xm.mark_step()
    _ = C.cpu()
    print("Bench setup passed")
