import torch

def run_test(helper_func):
    """Property-based test for gpu_tensor_hash.

    ``helper_func(tensor)`` must accept a CPU tensor and return a Python int.
    """
    # 1. Determinism
    torch.manual_seed(0)
    x = torch.randn(64, 64)
    h1 = helper_func(x)
    h2 = helper_func(x)
    assert h1 == h2, "Determinism: same input must produce same hash"

    # 2. Content sensitivity: flip one element
    x2 = x.clone()
    x2[0, 0] += 1.0
    assert helper_func(x) != helper_func(x2), "Hash must change when content changes"

    # 3. Empty tensor → 0
    assert helper_func(torch.empty(0)) == 0, "Empty tensor must hash to 0"

    # 4. Shape invariance: same bytes, different shape
    x = torch.arange(128, dtype=torch.float32)
    h_1d = helper_func(x)
    h_2d = helper_func(x.view(8, 16))
    assert h_1d == h_2d, "Same bytes in different shapes must hash identically"

    # 5. Non-collision: different random tensors hash differently
    a = torch.randn(32, 32)
    b = torch.randn(32, 32)
    assert helper_func(a) != helper_func(b), "Distinct tensors should hash differently"

def bench(kernel_fn):
    """Generate canonical benchmark inputs and invoke kernel_fn once.

    Uses a (1024, 1024) float32 tensor — exercises the hash kernel at a
    realistic model-weight inspection size.
    """
    x = torch.randn((1024, 1024))
    return kernel_fn(x)


# (rows, cols) — small / medium (matches bench()) / large
_BENCH3_SHAPES = [(512, 1024), (1024, 1024), (2048, 1024)]


def bench3(kernel_fn, idx):
    """Bench shape #idx (0..2) — 3-shape average variant of bench()."""
    x = torch.randn(_BENCH3_SHAPES[idx])
    return kernel_fn(x)
