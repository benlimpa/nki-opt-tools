"""
Shared oracle for opt10 sweep variants.

Reads P, N_total from the kernel module's globals. Tests row-wise
softmax(x) over the N dimension.
"""
import torch
import torch_xla.core.xla_model as xm


def _make_inputs(device):
    P_, N_ = int(P), int(N_total)  # noqa: F821 — read from kernel module
    torch.manual_seed(0)
    X = torch.randn(P_, N_, dtype=torch.float32)
    return X, X.to(device)


def run_test(wrapper):
    device = xm.xla_device()
    X, X_dev = _make_inputs(device)
    Y = wrapper(X_dev)
    xm.mark_step()
    Y_cpu = Y.cpu()
    ref = torch.softmax(X, dim=1)
    torch.testing.assert_close(Y_cpu, ref, atol=1e-5, rtol=1e-5)
    print("Test passed")


def bench(wrapper):
    device = xm.xla_device()
    _, X_dev = _make_inputs(device)
    Y = wrapper(X_dev)
    xm.mark_step()
    _ = Y.cpu()  # force XLA materialization → NEFF emission
    print("Bench setup passed")
