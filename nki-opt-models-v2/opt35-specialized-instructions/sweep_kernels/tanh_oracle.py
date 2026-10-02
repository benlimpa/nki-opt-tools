"""
Shared oracle for opt35 sweep variants.

Reads P, N from the kernel module's globals. Tests tanh(X * scale + bias)
when OPS=2, or tanh(X * scale) when OPS=1. nki-profile concatenates kernel
and oracle into one Python script before exec, so the kernel's top-level
constants are visible here.
"""
import torch
import torch_xla.core.xla_model as xm


def _make_inputs(device):
    P_, N_, OPS_ = int(P), int(N), int(OPS)  # noqa: F821
    torch.manual_seed(0)
    X = torch.randn(P_, N_, dtype=torch.float32)
    S = torch.randn(P_, 1, dtype=torch.float32)
    B = torch.randn(P_, 1, dtype=torch.float32) if OPS_ == 2 else torch.zeros(P_, 1, dtype=torch.float32)
    return X, S, B, X.to(device), S.to(device), B.to(device)


def run_test(wrapper):
    device = xm.xla_device()
    X, S, B, X_dev, S_dev, B_dev = _make_inputs(device)
    Y = wrapper(X_dev, S_dev, B_dev)
    xm.mark_step()
    Y_cpu = Y.cpu()
    if int(OPS) == 2:  # noqa: F821
        ref = torch.tanh(X * S + B)
    else:
        ref = torch.tanh(X * S)
    torch.testing.assert_close(Y_cpu, ref, atol=1e-3, rtol=1e-3)
    print("Test passed")


def bench(wrapper):
    device = xm.xla_device()
    _, _, _, X_dev, S_dev, B_dev = _make_inputs(device)
    Y = wrapper(X_dev, S_dev, B_dev)
    xm.mark_step()
    _ = Y.cpu()  # force XLA materialization → NEFF emission
    print("Bench setup passed")
