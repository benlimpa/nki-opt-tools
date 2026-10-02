import torch

def _ref_cumsum(g: torch.Tensor, chunk_size: int, reverse: bool = False, scale: float = None) -> torch.Tensor:
    """Pure-PyTorch reference for chunk-local cumsum."""
    B, H, T = g.shape
    g_flat = g.reshape(B * H, -1)
    T_pad = ((T + chunk_size - 1) // chunk_size) * chunk_size
    g_padded = torch.zeros((B * H, T_pad), device=g.device, dtype=g.dtype)
    g_padded[:, :T] = g_flat

    g_chunks = g_padded.view(B * H, -1, chunk_size)

    if reverse:
        # Reverse cumsum: sum(x[i:])
        # Can be done as total_sum - forward_cumsum + current_val
        out_chunks = torch.flip(torch.cumsum(torch.flip(g_chunks, dims=[-1]), dim=-1), dims=[-1])
    else:
        out_chunks = torch.cumsum(g_chunks, dim=-1)

    if scale is not None:
        out_chunks = out_chunks * scale

    out_flat = out_chunks.view(B * H, -1)[:, :T]
    return out_flat.view(B, H, T)

def run_test(helper_func):
    """Property-based test for the cumsum kernel."""
    # 1. Differential test
    for B, H, T, chunk_size in [(2, 4, 128, 32), (1, 8, 256, 64), (4, 2, 64, 64)]:
        x = torch.randn((B, H, T))
        expected = _ref_cumsum(x, chunk_size)
        actual = helper_func(x, chunk_size)
        torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=1e-4, rtol=1e-4)

def bench(kernel_fn):
    """Generate canonical benchmark inputs and invoke kernel_fn once.

    Uses (4, 8, 2048) with chunk_size=64: batch=4, heads=8, seq=2048 —
    representative of a chunk-wise state-space model cumsum pass.
    """
    x = torch.randn((4, 8, 2048))
    return kernel_fn(x, 64)


# (B, H, T) — small / medium (matches bench()) / large; chunk_size=64
_BENCH3_SHAPES = [(2, 8, 2048), (4, 8, 2048), (8, 8, 2048)]


def bench3(kernel_fn, idx):
    """Bench shape #idx (0..2) — 3-shape average variant of bench()."""
    x = torch.randn(_BENCH3_SHAPES[idx])
    return kernel_fn(x, 64)
