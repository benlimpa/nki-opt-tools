"""
Shared oracle for opt37 sweep variants.

Reads D0, D1, D2, D3 from the kernel module's globals. Tests
``X.transpose(0, 3)`` over a 4D input where D1=D2=1.
"""
import torch
import torch_xla.core.xla_model as xm


def _make_inputs(device):
    D0_, D1_, D2_, D3_ = int(D0), int(D1), int(D2), int(D3)  # noqa: F821
    torch.manual_seed(0)
    X = torch.randn(D0_, D1_, D2_, D3_, dtype=torch.float32)
    return X, X.to(device)


def run_test(wrapper):
    device = xm.xla_device()
    X, X_dev = _make_inputs(device)
    Y = wrapper(X_dev)
    xm.mark_step()
    Y_cpu = Y.cpu()
    ref = X.transpose(0, 3)
    torch.testing.assert_close(Y_cpu, ref, atol=1e-3, rtol=1e-3)
    print("Test passed")


def bench(wrapper):
    device = xm.xla_device()
    _, X_dev = _make_inputs(device)
    Y = wrapper(X_dev)
    xm.mark_step()
    _ = Y.cpu()  # force XLA materialization → NEFF emission
    print("Bench setup passed")
