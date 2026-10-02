import torch

def _ref_sparsemax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Reference sparsemax implementation in pure PyTorch (CPU-friendly)."""
    x = x.float()
    d = x.shape[dim]
    sorted_x, _ = torch.sort(x, dim=dim, descending=True)

    # Broadcast-friendly range 1..d along `dim`
    shape = [1] * x.dim()
    shape[dim] = d
    r = torch.arange(1, d + 1, dtype=torch.float32, device=x.device).view(shape)

    cssv = torch.cumsum(sorted_x, dim=dim)
    support = sorted_x > (cssv - 1.0) / r
    k = support.sum(dim=dim, keepdim=True).float().clamp(min=1)
    s = (sorted_x * support.float()).sum(dim=dim, keepdim=True)
    tau = (s - 1.0) / k
    return torch.clamp(x - tau, min=0.0)

def run_test(helper_func):
    """Property-based test for the sparsemax kernel.

    ``helper_func(x)`` must accept a CPU tensor and return a tensor
    (possibly on device). Comparison done after .cpu().
    """
    # 1. Differential test – several shapes
    for rows, cols in [(1, 32), (4, 64), (16, 128), (32, 256)]:
        x = torch.randn((rows, cols))
        expected = _ref_sparsemax(x)
        actual = helper_func(x)
        torch.testing.assert_close(actual.cpu(), expected, atol=1e-4, rtol=1e-4)

    # 2. Non-negativity
    x = torch.randn((32, 128))
    out = helper_func(x).cpu()
    assert (out >= -1e-6).all(), "sparsemax output must be non-negative"

    # 3. Sum-to-one
    x = torch.randn((32, 128))
    out = helper_func(x).cpu()
    row_sums = out.sum(dim=-1)
    torch.testing.assert_close(row_sums, torch.ones(32), atol=1e-4, rtol=1e-4)

    # 4. Sparsity: peaky input should yield at least one zero per row
    x = torch.zeros((8, 64))
    x[:, 0] = 10.0  # dominant entry
    out = helper_func(x).cpu()
    assert (out[:, 1:] == 0.0).all(), "All non-dominant entries should be zero"

    # 5. Idempotency: sparsemax(sparsemax(x)) == sparsemax(x)
    x = torch.randn((8, 64))
    p = helper_func(x).cpu()
    pp = helper_func(p)
    torch.testing.assert_close(pp.cpu(), p, atol=1e-5, rtol=1e-5)

    # 6. Numerical stability – large-magnitude inputs
    x_large = torch.randn((8, 128)) * 100.0
    out = helper_func(x_large).cpu()
    assert torch.isfinite(out).all(), "Output must be finite for large inputs"
    row_sums = out.sum(dim=-1)
    torch.testing.assert_close(row_sums, torch.ones(8), atol=1e-4, rtol=1e-4)

    # 7. Determinism
    torch.manual_seed(42)
    x = torch.randn((16, 128))
    out1 = helper_func(x).cpu()
    out2 = helper_func(x).cpu()
    torch.testing.assert_close(out1, out2)

def bench(kernel_fn):
    """Generate canonical benchmark inputs and invoke kernel_fn once.

    Uses (512, 4096) — representative of a large-vocabulary sparse attention
    distribution computation.
    """
    x = torch.randn((512, 4096))
    return kernel_fn(x)


# (rows, cols) — small / medium (matches bench()) / large
_BENCH3_SHAPES = [(256, 4096), (512, 4096), (1024, 4096)]


def bench3(kernel_fn, idx):
    """Bench shape #idx (0..2) — 3-shape average variant of bench()."""
    x = torch.randn(_BENCH3_SHAPES[idx])
    return kernel_fn(x)
