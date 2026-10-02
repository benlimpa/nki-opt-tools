"""
Shared oracle for opt21 sweep variants (row-wise mean subtraction).

Reads M, N from kernel module globals. Tests y[m,n] = x[m,n] - mean(x[m,:]).
"""
import torch
import torch_xla.core.xla_model as xm


def _make_inputs(device):
    M_, N_ = int(M), int(N)  # noqa: F821 — read from kernel module
    torch.manual_seed(0)
    x = torch.randn(M_, N_, dtype=torch.float32)
    return x, x.to(device)


def run_test(wrapper):
    device = xm.xla_device()
    x, x_dev = _make_inputs(device)
    y = wrapper(x_dev)
    xm.mark_step()
    y_cpu = y.cpu()
    ref = x - x.mean(dim=1, keepdim=True)
    torch.testing.assert_close(y_cpu, ref, atol=1e-3, rtol=1e-4)
    print("Test passed")


def bench(wrapper):
    device = xm.xla_device()
    _, x_dev = _make_inputs(device)
    y = wrapper(x_dev)
    xm.mark_step()
    _ = y.cpu()
    print("Bench setup passed")
