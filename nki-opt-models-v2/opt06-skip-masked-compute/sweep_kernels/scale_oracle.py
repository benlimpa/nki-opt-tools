"""
Shared oracle for opt06 sweep variants.

Reads M, N, TILE_P, VALID_TILES from kernel module globals. Tests
y = x * 2 over the first VALID_TILES * TILE_P rows; padding tiles are
ignored (their output value is irrelevant since the optimization skips
them).
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
    valid_rows = int(VALID_TILES) * int(TILE_P)  # noqa: F821
    ref = X[:valid_rows] * 2.0
    torch.testing.assert_close(Y_cpu[:valid_rows], ref, atol=1e-5, rtol=1e-5)
    print("Test passed")


def bench(wrapper):
    device = xm.xla_device()
    _, X_dev = _make_inputs(device)
    Y = wrapper(X_dev)
    xm.mark_step()
    _ = Y.cpu()  # force XLA materialization → NEFF emission
    print("Bench setup passed")
