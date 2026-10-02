"""
Shared oracle for opt07 sweep variants.

Reads M, N, TILE_P, N_OPS from the kernel module's globals. Each variant
applies a deterministic chain of N_OPS scalar ops to compute the
reference output. The before/after kernels apply the same chain;
only the loop fusion structure differs.

Op chain (alternating mul/add): for i=0..N_OPS-1:
  if i even: y *= (2.0 - 0.5 * i)        # 2.0, 1.0, 0.0 ...
  if i odd:  y += (1.0 - 0.5 * (i - 1))  # 1.0, 0.5 ...
For N_OPS=2: y = x * 2 + 1
For N_OPS=3: y = (x * 2 + 1) * 0.5
For N_OPS=4: y = ((x * 2 + 1) * 0.5) + 0.0
"""
import torch
import torch_xla.core.xla_model as xm


def _make_inputs(device):
    M_, N_ = int(M), int(N)  # noqa: F821 — read from kernel module
    torch.manual_seed(0)
    X = torch.randn(M_, N_, dtype=torch.float32)
    return X, X.to(device)


def _ref_output(X):
    N_OPS_ = int(N_OPS)  # noqa: F821 — read from kernel module
    y = X.clone()
    for i in range(N_OPS_):
        if i % 2 == 0:
            y = y * (2.0 - 0.5 * i)
        else:
            y = y + (1.0 - 0.5 * (i - 1))
    return y


def run_test(wrapper):
    device = xm.xla_device()
    X, X_dev = _make_inputs(device)
    Y = wrapper(X_dev)
    xm.mark_step()
    Y_cpu = Y.cpu()
    ref = _ref_output(X)
    torch.testing.assert_close(Y_cpu, ref, atol=1e-3, rtol=1e-3)
    print("Test passed")


def bench(wrapper):
    device = xm.xla_device()
    _, X_dev = _make_inputs(device)
    Y = wrapper(X_dev)
    xm.mark_step()
    _ = Y.cpu()  # force XLA materialization → NEFF emission
    print("Bench setup passed")
