"""
Oracle for opt38 before.py / after.py.

Provides:
  - run_test(wrapper): correctness check (used by nki-profile default mode)
  - bench(wrapper):    benchmark-shaped invocation (used by nki-profile --bench)

The wrapper closes over M and N — defined in the kernel module's globals
when this oracle is concatenated with the kernel by the profiler.
"""
import torch
import torch_xla.core.xla_model as xm


def _make_inputs(device):
    M_, N_ = int(M), int(N)  # noqa: F821 — read from kernel module
    torch.manual_seed(0)
    x = torch.randn(M_, N_, dtype=torch.float32)
    y = torch.randn(M_, N_, dtype=torch.float32)
    return x, y, x.to(device), y.to(device)


def run_test(wrapper):
    device = xm.xla_device()
    x, y, x_dev, y_dev = _make_inputs(device)
    z = wrapper(x_dev, y_dev)
    xm.mark_step()
    z_cpu = z.cpu()
    ref = x + y
    torch.testing.assert_close(z_cpu, ref, atol=1e-5, rtol=1e-5)
    print("Test passed")


def bench(wrapper):
    device = xm.xla_device()
    _, _, x_dev, y_dev = _make_inputs(device)
    z = wrapper(x_dev, y_dev)
    xm.mark_step()
    _ = z.cpu()
    print("Bench setup passed")
