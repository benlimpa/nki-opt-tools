"""
Shared oracle for opt24 sweep variants.

Reads P, N from the kernel module's globals. Tests the polynomial map
y = 0.79788 * (x + 0.044715 * x^3) on a [P, N] bf16 input.
"""
import torch
import torch_xla.core.xla_model as xm


def _make_inputs(device):
    P_, N_ = int(P), int(N)  # noqa: F821 — read from kernel module
    torch.manual_seed(0)
    X = torch.randn(P_, N_, dtype=torch.bfloat16)
    return X, X.to(device)


def run_test(wrapper):
    device = xm.xla_device()
    X, X_dev = _make_inputs(device)
    Y = wrapper(X_dev)
    xm.mark_step()
    Y_cpu = Y.cpu()
    ref_x = X.float()
    ref_y = (0.79788 * (ref_x + 0.044715 * ref_x.pow(3))).bfloat16()
    torch.testing.assert_close(Y_cpu.float(), ref_y.float(), atol=1e-1, rtol=1e-2)
    print("Test passed")


def bench(wrapper):
    device = xm.xla_device()
    _, X_dev = _make_inputs(device)
    Y = wrapper(X_dev)
    xm.mark_step()
    _ = Y.cpu()  # force XLA materialization → NEFF emission
    print("Bench setup passed")
