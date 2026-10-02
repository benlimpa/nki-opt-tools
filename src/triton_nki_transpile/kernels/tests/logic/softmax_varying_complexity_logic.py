import torch

def _ref_softmax(x: torch.Tensor) -> torch.Tensor:
    """Row-wise softmax reference computed in float32."""
    return torch.softmax(x.float(), dim=-1)

_COMMON_SHAPES = [(1, 64), (4, 128), (8, 256), (16, 512), (32, 1024)]

def run_test(helper_func):
    """Property-based test for a row-wise softmax kernel.

    ``helper_func(x)`` must accept a CPU float32 tensor of shape (M, N)
    with N a power-of-2 in [1, 4096] and return a tensor (possibly on
    device).  Comparison is performed after ``.cpu()``.
    """
    # 1. Differential — several common shapes
    for m, n in _COMMON_SHAPES:
        x = torch.randn((m, n))
        torch.testing.assert_close(
            helper_func(x).cpu(), _ref_softmax(x), atol=1e-5, rtol=1e-5
        )

    # 2. Rows sum to 1
    m, n = 16, 256
    x = torch.randn((m, n))
    out = helper_func(x).cpu()
    torch.testing.assert_close(
        out.sum(dim=-1), torch.ones(m), atol=1e-5, rtol=1e-5
    )

    # 3. Non-negative
    assert (out >= 0).all(), "softmax output must be non-negative"

    # 4. Small-shift invariance (safe for naive's no max-subtract)
    torch.manual_seed(7)
    x = torch.randn((8, 128))
    out_base    = helper_func(x).cpu()
    out_shifted = helper_func(x + 10.0).cpu()
    torch.testing.assert_close(out_shifted, out_base, atol=1e-5, rtol=1e-5)

    # 5. Uniform input → uniform output
    m, n = 4, 128
    out_unif = helper_func(torch.zeros((m, n))).cpu()
    torch.testing.assert_close(
        out_unif, torch.full((m, n), 1.0 / n), atol=1e-5, rtol=1e-5
    )

    # 6. Argmax preserved
    torch.manual_seed(13)
    x = torch.randn((16, 256))
    out = helper_func(x).cpu()
    assert (out.argmax(dim=-1) == x.argmax(dim=-1)).all()

    # 7. Determinism
    torch.manual_seed(99)
    x = torch.randn((16, 512))
    out1 = helper_func(x).cpu()
    out2 = helper_func(x).cpu()
    torch.testing.assert_close(out1, out2)

def bench(kernel_fn):
    """Generate canonical benchmark inputs and invoke kernel_fn once.

    Uses (512, 4096) float32 — representative of a large-sequence row-wise
    softmax over attention logits.
    """
    x = torch.randn((512, 4096))
    return kernel_fn(x)


# (M, N) — small / medium (matches bench()) / large; N must be power-of-2 in [1, 4096]
_BENCH3_SHAPES = [(256, 4096), (512, 4096), (1024, 4096)]


def bench3(kernel_fn, idx):
    """Bench shape #idx (0..2) — 3-shape average variant of bench()."""
    x = torch.randn(_BENCH3_SHAPES[idx])
    return kernel_fn(x)
