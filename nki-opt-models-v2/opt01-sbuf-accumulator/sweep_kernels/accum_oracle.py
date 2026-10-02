"""
Shared oracle for opt01 sweep variants.

Reads M, N, TILE_M from the kernel module's globals. Tests the column-wise
partial reduction y[j, k] = sum_{i} x[i*TILE_M + j, k] for an [M, N] input.
"""
import torch
import torch_xla.core.xla_model as xm


def _make_inputs(device):
    M_, N_ = int(M), int(N)  # noqa: F821 — read from kernel module
    torch.manual_seed(0)
    X = torch.randn(M_, N_, dtype=torch.float32)
    return X, X.to(device)


def run_test(wrapper):
    device = xm.xla_device()
    X, X_dev = _make_inputs(device)
    Y = wrapper(X_dev)
    xm.mark_step()
    Y_cpu = Y.cpu()
    M_, N_, TILE_M_ = int(M), int(N), int(TILE_M)  # noqa: F821
    ref = torch.zeros(TILE_M_, N_, dtype=torch.float32)
    for i in range(M_ // TILE_M_):
        ref += X[i * TILE_M_:(i + 1) * TILE_M_, :]
    torch.testing.assert_close(Y_cpu, ref, atol=1e-3, rtol=1e-3)
    print("Test passed")


def bench(wrapper):
    device = xm.xla_device()
    _, X_dev = _make_inputs(device)
    Y = wrapper(X_dev)
    xm.mark_step()
    _ = Y.cpu()  # force XLA materialization → NEFF emission
    print("Bench setup passed")
