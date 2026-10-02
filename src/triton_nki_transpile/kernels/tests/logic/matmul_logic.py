import torch

def _ref_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """PyTorch float32 reference, cast back to float16 to match kernel output."""
    return torch.matmul(a.float(), b.float()).half()

def run_test(helper_func):
    """Property-based test for the matmul kernel.

    ``helper_func(a, b)`` must accept float16 CPU tensors and return a
    float16 tensor (possibly on a device). Comparison is done after .cpu().
    """
    # 1. Differential test across several shapes (powers of two)
    shapes = [
        (64, 64, 64),
        (128, 128, 128),
        (256, 128, 64),
        (64, 256, 128),
        (128, 64, 256),
    ]
    for m, n, k in shapes:
        a = torch.randn((m, k), dtype=torch.float16)
        b = torch.randn((k, n), dtype=torch.float16)
        expected = _ref_matmul(a, b)
        actual = helper_func(a, b)
        torch.testing.assert_close(actual.cpu(), expected, atol=1e-2, rtol=1e-2)

    # 2. Non-power-of-two shapes
    for m, n, k in [(96, 80, 48), (100, 120, 80), (33, 65, 97)]:
        a = torch.randn((m, k), dtype=torch.float16)
        b = torch.randn((k, n), dtype=torch.float16)
        expected = _ref_matmul(a, b)
        actual = helper_func(a, b)
        torch.testing.assert_close(actual.cpu(), expected, atol=1e-2, rtol=1e-2)

    # 3. Metamorphic – scaling A by scalar c scales output by c
    a = torch.randn((128, 64), dtype=torch.float16)
    b = torch.randn((64, 128), dtype=torch.float16)
    out_base = helper_func(a, b).cpu().float()
    for scale in [2.0, 0.5]:
        out_scaled = helper_func((a * scale).half(), b).cpu().float()
        torch.testing.assert_close(out_scaled, out_base * scale, atol=5e-2, rtol=5e-2)

    # 4. Identity matrix: A @ I == A (within float16 precision)
    n = 128
    a = torch.randn((n, n), dtype=torch.float16)
    eye = torch.eye(n, dtype=torch.float16)
    result = helper_func(a, eye).cpu()
    torch.testing.assert_close(result.float(), a.float(), atol=5e-2, rtol=5e-2)

    # 5. Numerical stability – small magnitude inputs
    a = torch.full((64, 64), 1e-3, dtype=torch.float16)
    b = torch.full((64, 64), 1e-3, dtype=torch.float16)
    expected = _ref_matmul(a, b)
    actual = helper_func(a, b)
    torch.testing.assert_close(actual.cpu(), expected, atol=1e-4, rtol=1e-1)

    # 6. Determinism – same inputs produce identical outputs
    torch.manual_seed(42)
    a = torch.randn((128, 128), dtype=torch.float16)
    b = torch.randn((128, 128), dtype=torch.float16)
    out1 = helper_func(a, b).cpu()
    out2 = helper_func(a, b).cpu()
    torch.testing.assert_close(out1, out2)

def bench(kernel_fn):
    """Generate canonical benchmark inputs and invoke kernel_fn once.

    Uses a 2048×2048 float16 matmul, representative of a transformer MLP layer.
    """
    a = torch.randn((2048, 2048), dtype=torch.float16)
    b = torch.randn((2048, 2048), dtype=torch.float16)
    return kernel_fn(a, b)


# (M, K, N) — small / medium (matches bench()) / large; varies M
_BENCH3_SHAPES = [(1024, 2048, 2048), (2048, 2048, 2048), (4096, 2048, 2048)]


def bench3(kernel_fn, idx):
    """Bench shape #idx (0..2) — 3-shape average variant of bench()."""
    M, K, N = _BENCH3_SHAPES[idx]
    a = torch.randn((M, K), dtype=torch.float16)
    b = torch.randn((K, N), dtype=torch.float16)
    return kernel_fn(a, b)
