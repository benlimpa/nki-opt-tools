"""
Shared oracle for opt18 sweep variants.

Reads M, K, N from the kernel module's globals. Tests
``C = lhsT.T @ rhs`` where ``lhsT [K, M]``, ``rhs [K, N]``.
"""
import torch
import torch_xla.core.xla_model as xm


def _make_inputs(device):
    M_, K_, N_ = int(M), int(K), int(N)  # noqa: F821
    torch.manual_seed(0)
    lhsT = torch.randn(K_, M_, dtype=torch.float32)
    rhs = torch.randn(K_, N_, dtype=torch.float32)
    return lhsT, rhs, lhsT.to(device), rhs.to(device)


def run_test(wrapper):
    device = xm.xla_device()
    lhsT, rhs, lhsT_dev, rhs_dev = _make_inputs(device)
    C = wrapper(lhsT_dev, rhs_dev)
    xm.mark_step()
    C_cpu = C.cpu()
    ref = lhsT.T @ rhs
    torch.testing.assert_close(C_cpu, ref, atol=1e-2, rtol=1e-2)
    print("Test passed")


def bench(wrapper):
    device = xm.xla_device()
    _, _, lhsT_dev, rhs_dev = _make_inputs(device)
    C = wrapper(lhsT_dev, rhs_dev)
    xm.mark_step()
    _ = C.cpu()  # force XLA materialization → NEFF emission
    print("Bench setup passed")
