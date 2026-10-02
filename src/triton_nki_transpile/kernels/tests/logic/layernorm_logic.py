import torch
import torch.nn.functional as F

def _ref_layer_norm(x, weight, bias, eps):
    """PyTorch reference implementation."""
    N = x.shape[-1]
    return F.layer_norm(x.float(), [N], weight.float(), bias.float(), eps).to(x.dtype)

def run_test(helper_func):
    """Property-based test for the layer_norm kernel.

    ``helper_func(x, weight, bias, eps)`` must accept CPU tensors and return
    a tensor (possibly on a device). Comparison done after .cpu().
    """
    eps = 1e-5

    # 1. Differential test – several 2-D shapes
    for m, n in [(1, 128), (4, 256), (16, 512), (32, 64)]:
        x = torch.randn((m, n), dtype=torch.float32)
        w = torch.randn((n,), dtype=torch.float32)
        b = torch.randn((n,), dtype=torch.float32)
        expected = _ref_layer_norm(x, w, b, eps)
        actual = helper_func(x, w, b, eps)
        torch.testing.assert_close(actual.cpu(), expected, atol=1e-3, rtol=1e-3)

    # 2. 3-D input (batch × seq × features) – kernel reshapes internally
    x = torch.randn((4, 8, 64), dtype=torch.float32)
    w = torch.randn((64,), dtype=torch.float32)
    b = torch.randn((64,), dtype=torch.float32)
    expected = _ref_layer_norm(x, w, b, eps)
    actual = helper_func(x, w, b, eps)
    torch.testing.assert_close(actual.cpu(), expected, atol=1e-3, rtol=1e-3)

    # 3. Statistical invariant: with w=1, b=0 the output per-row has mean≈0, std≈1
    m, n = 16, 256
    x = torch.randn((m, n), dtype=torch.float32)
    w = torch.ones((n,), dtype=torch.float32)
    b = torch.zeros((n,), dtype=torch.float32)
    out = helper_func(x, w, b, eps).cpu()
    torch.testing.assert_close(out.mean(dim=-1), torch.zeros(m), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(out.std(dim=-1), torch.ones(m), atol=1e-2, rtol=1e-2)

    # 4. Algebraic: zero bias → output is scale * norm; zero weight → output is all zeros
    x = torch.randn((8, 128), dtype=torch.float32)
    w = torch.zeros((128,), dtype=torch.float32)
    b_zero = torch.zeros((128,), dtype=torch.float32)
    out_zero_w = helper_func(x, w, b_zero, eps).cpu()
    torch.testing.assert_close(out_zero_w, torch.zeros_like(out_zero_w), atol=1e-5, rtol=0)

    # 5. Numerical stability – large magnitude inputs
    x_large = torch.full((4, 128), 1e4, dtype=torch.float32) + torch.randn((4, 128)) * 1e3
    w = torch.ones((128,), dtype=torch.float32)
    b = torch.zeros((128,), dtype=torch.float32)
    expected = _ref_layer_norm(x_large, w, b, eps)
    actual = helper_func(x_large, w, b, eps)
    torch.testing.assert_close(actual.cpu(), expected, atol=1e-3, rtol=1e-3)

    # 6. Determinism
    x = torch.randn((8, 128), dtype=torch.float32)
    w = torch.randn((128,), dtype=torch.float32)
    b = torch.randn((128,), dtype=torch.float32)
    out1 = helper_func(x, w, b, eps).cpu()
    out2 = helper_func(x, w, b, eps).cpu()
    torch.testing.assert_close(out1, out2)

def bench(kernel_fn):
    """Generate canonical benchmark inputs and invoke kernel_fn once.

    Uses a (512, 2048) activation matrix, representative of a transformer layer norm.
    """
    x = torch.randn((512, 2048), dtype=torch.float32)
    w = torch.ones((2048,), dtype=torch.float32)
    b = torch.zeros((2048,), dtype=torch.float32)
    return kernel_fn(x, w, b, 1e-5)


# (rows, cols) — small / medium (matches bench()) / large
_BENCH3_SHAPES = [(256, 2048), (512, 2048), (1024, 2048)]


def bench3(kernel_fn, idx):
    """Bench shape #idx (0..2) — 3-shape average variant of bench()."""
    rows, cols = _BENCH3_SHAPES[idx]
    x = torch.randn((rows, cols), dtype=torch.float32)
    w = torch.ones((cols,), dtype=torch.float32)
    b = torch.zeros((cols,), dtype=torch.float32)
    return kernel_fn(x, w, b, 1e-5)
