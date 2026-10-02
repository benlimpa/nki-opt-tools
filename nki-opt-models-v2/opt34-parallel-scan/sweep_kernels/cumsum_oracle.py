"""
Shared oracle for opt34 sweep variants.

Reads N_TOTAL, P from the kernel module's globals. Tests 1D cumsum over
N_TOTAL elements; passes both X and L_tri (a [P, P] strictly upper-triangular
matrix of ones) to the kernel — the BEFORE kernel ignores L_tri.
"""
import torch
import torch_xla.core.xla_model as xm


def _make_inputs(device):
    N_, P_ = int(N_TOTAL), int(P)  # noqa: F821
    torch.manual_seed(0)
    X = torch.randn(N_, dtype=torch.float32)
    L_tri = torch.triu(torch.ones(P_, P_), diagonal=1)
    return X, L_tri, X.to(device), L_tri.to(device)


def run_test(wrapper):
    device = xm.xla_device()
    X, L_tri, X_dev, L_tri_dev = _make_inputs(device)
    Y = wrapper(X_dev, L_tri_dev)
    xm.mark_step()
    Y_cpu = Y.cpu()
    ref = torch.cumsum(X, dim=0)
    torch.testing.assert_close(Y_cpu, ref, atol=1e-2, rtol=1e-2)
    print("Test passed")


def bench(wrapper):
    device = xm.xla_device()
    _, _, X_dev, L_tri_dev = _make_inputs(device)
    Y = wrapper(X_dev, L_tri_dev)
    xm.mark_step()
    _ = Y.cpu()  # force XLA materialization → NEFF emission
    print("Bench setup passed")
