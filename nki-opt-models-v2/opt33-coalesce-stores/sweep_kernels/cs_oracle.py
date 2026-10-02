"""
Shared oracle for opt33 sweep variants.

Reads N_ITERS, N_FRAG, TILE_M, FRAG_W from kernel module globals.
Tests y = 2.0 * x for x of shape [TILE_M, N_ITERS * N_FRAG * FRAG_W].
"""
import torch
import torch_xla.core.xla_model as xm


def _make_inputs(device):
    M_ = int(TILE_M)            # noqa: F821
    W_ = int(N_ITERS) * int(N_FRAG) * int(FRAG_W)  # noqa: F821
    torch.manual_seed(0)
    x = torch.randn(M_, W_, dtype=torch.float32)
    return x, x.to(device)


def run_test(wrapper):
    device = xm.xla_device()
    x, x_dev = _make_inputs(device)
    y = wrapper(x_dev)
    xm.mark_step()
    y_cpu = y.cpu()
    ref = x * 2.0
    torch.testing.assert_close(y_cpu, ref, atol=1e-3, rtol=1e-4)
    print("Test passed")


def bench(wrapper):
    device = xm.xla_device()
    _, x_dev = _make_inputs(device)
    y = wrapper(x_dev)
    xm.mark_step()
    _ = y.cpu()
    print("Bench setup passed")
