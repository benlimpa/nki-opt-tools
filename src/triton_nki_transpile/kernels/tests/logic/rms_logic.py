import torch

def _ref_rms_layernorm(X: torch.Tensor, W: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Manual PyTorch reference for RMS LayerNorm.

    Y[i] = (X[i] / sqrt(mean(X[i]^2) + eps)) * W   (dtype matches X)
    """
    X_f = X.float()
    rms = torch.sqrt((X_f ** 2).mean(dim=-1, keepdim=True) + eps)
    normed = (X_f / rms).to(X.dtype)
    return normed * W.to(X.dtype)

def run_test(helper_func):
    """Property-based test for the rms_layernorm kernel.

    ``helper_func(X, W, eps)`` must accept CPU tensors and return a tensor
    (possibly on device). Comparison done after .cpu().
    """
    eps = 1e-5

    # 1. Differential test – power-of-two column counts
    for rows, cols in [(1, 64), (4, 128), (16, 256), (32, 512), (8, 64)]:
        X = torch.randn((rows, cols))
        W = torch.randn((cols,))
        expected = _ref_rms_layernorm(X, W, eps)
        actual = helper_func(X, W, eps)
        torch.testing.assert_close(actual.cpu(), expected, atol=1e-3, rtol=1e-3)

    # 2. Unit weight: output norms equal inv_rms * ||X_row||  (≈ sqrt(cols))
    rows, cols = 16, 128
    X = torch.randn((rows, cols))
    W = torch.ones((cols,))
    out = helper_func(X, W, eps).cpu().float()
    # The RMS of each output row with unit W should be 1 (by definition of RMS norm)
    out_rms = torch.sqrt((out ** 2).mean(dim=-1))
    torch.testing.assert_close(out_rms, torch.ones(rows), atol=1e-2, rtol=1e-2)

    # 3. Algebraic – zero weight forces output to zero
    X = torch.randn((8, 64))
    W = torch.zeros((64,))
    out = helper_func(X, W, eps).cpu()
    torch.testing.assert_close(out, torch.zeros_like(out), atol=1e-6, rtol=0)

    # 4. Metamorphic – scaling X by c does NOT change output (RMS norm is scale-invariant)
    X = torch.randn((8, 64))
    W = torch.randn((64,))
    out_base = helper_func(X, W, eps).cpu()
    for scale in [2.0, 0.5, 10.0]:
        out_scaled = helper_func(X * scale, W, eps).cpu()
        torch.testing.assert_close(out_scaled, out_base, atol=1e-3, rtol=1e-3)

    # 5. Numerical stability – near-zero inputs
    X_small = torch.full((4, 64), 1e-7) + torch.randn((4, 64)) * 1e-7
    W = torch.ones((64,))
    out = helper_func(X_small, W, eps).cpu()
    assert torch.isfinite(out).all(), "Output must be finite for near-zero inputs"

    # 6. Determinism
    torch.manual_seed(99)
    X = torch.randn((16, 128))
    W = torch.randn((128,))
    out1 = helper_func(X, W, eps).cpu()
    out2 = helper_func(X, W, eps).cpu()
    torch.testing.assert_close(out1, out2)

def bench(kernel_fn):
    """Generate canonical benchmark inputs and invoke kernel_fn once.

    Uses a (512, 2048) activation matrix, representative of a transformer RMS norm.
    """
    X = torch.randn((512, 2048))
    W = torch.ones((2048,))
    return kernel_fn(X, W, 1e-5)


# (rows, cols) — small / medium (matches bench()) / large
_BENCH3_SHAPES = [(256, 2048), (512, 2048), (1024, 2048)]


def bench3(kernel_fn, idx):
    """Bench shape #idx (0..2) — 3-shape average variant of bench()."""
    rows, cols = _BENCH3_SHAPES[idx]
    X = torch.randn((rows, cols))
    W = torch.ones((cols,))
    return kernel_fn(X, W, 1e-5)
