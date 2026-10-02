"""
Shared oracle for opt31 sweep variants.

Reads M, N from the kernel module's globals. Tests row-wise min-max
normalization: y[m, n] = (x[m, n] - row_min[m]) / (row_max[m] - row_min[m] + eps).
"""
import torch
import torch_xla.core.xla_model as xm

EPS = 1e-8


def _make_inputs(device):
    M_, N_ = int(M), int(N)  # noqa: F821
    torch.manual_seed(42)
    x = torch.randn(M_, N_, dtype=torch.float32)
    return x, x.to(device)


def run_test(wrapper):
    device = xm.xla_device()
    x, x_dev = _make_inputs(device)
    y = wrapper(x_dev)
    xm.mark_step()
    y_cpu = y.cpu()
    row_min = x.min(dim=1, keepdim=True).values
    row_max = x.max(dim=1, keepdim=True).values
    ref = (x - row_min) / (row_max - row_min + EPS)
    torch.testing.assert_close(y_cpu, ref, atol=1e-4, rtol=1e-4)
    print("Test passed")


def bench(wrapper):
    device = xm.xla_device()
    _, x_dev = _make_inputs(device)
    y = wrapper(x_dev)
    xm.mark_step()
    _ = y.cpu()  # force XLA materialization → NEFF emission
    print("Bench setup passed")
