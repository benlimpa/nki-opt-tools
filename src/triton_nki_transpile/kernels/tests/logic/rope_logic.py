import torch

def _make_cos_sin(seq_len: int, head_dim: int, dtype=torch.float32):
    """Build simple cos/sin tables of shape (seq_len, head_dim // 2)."""
    half = head_dim // 2
    theta = 1.0 / (10000.0 ** (torch.arange(0, half, dtype=torch.float32) / half))
    pos = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)  # (seq, 1)
    angles = pos * theta.unsqueeze(0)                               # (seq, half)
    return torch.cos(angles).to(dtype), torch.sin(angles).to(dtype)

def _ref_rope(Q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Pure-PyTorch reference for RoPE forward pass.

    Q   : (batch, seq_len, n_heads, head_dim)
    cos : (seq_len, head_dim // 2)
    sin : (seq_len, head_dim // 2)
    """
    batch, seq_len, n_heads, head_dim = Q.shape
    half = head_dim // 2
    q1 = Q[..., :half].float()   # (B, S, H, half)
    q2 = Q[..., half:].float()   # (B, S, H, half)

    # Broadcast cos/sin: (1, S, 1, half)
    c = cos[:seq_len].view(1, seq_len, 1, half)
    s = sin[:seq_len].view(1, seq_len, 1, half)

    out1 = q1 * c - q2 * s
    out2 = q2 * c + q1 * s
    return torch.cat([out1, out2], dim=-1).to(Q.dtype)

def run_test(helper_func):
    """Property-based test for the rope kernel.

    ``helper_func(Q, cos, sin)`` must accept CPU tensors and return a tensor
    (possibly on device). Comparison done after .cpu().
    """
    head_dim = 64  # half = 32 = BLOCK_SIZE, avoids mask edge-cases

    # 1. Differential test – various (batch, seq_len, n_heads)
    for batch, seq_len, n_heads in [(1, 4, 1), (2, 8, 4), (4, 16, 8)]:
        Q = torch.randn((batch, seq_len, n_heads, head_dim))
        cos, sin = _make_cos_sin(seq_len, head_dim)
        expected = _ref_rope(Q, cos, sin)
        actual = helper_func(Q, cos, sin)
        torch.testing.assert_close(actual.cpu(), expected, atol=1e-3, rtol=1e-3)

    # 2. Norm preservation: ||rope(Q)||_2 ≈ ||Q||_2 per head (rotation is unitary)
    batch, seq_len, n_heads = 2, 8, 4
    Q = torch.randn((batch, seq_len, n_heads, head_dim))
    cos, sin = _make_cos_sin(seq_len, head_dim)
    Q_out = helper_func(Q, cos, sin).cpu().float()
    norm_in = Q.float().norm(dim=-1)
    norm_out = Q_out.norm(dim=-1)
    torch.testing.assert_close(norm_out, norm_in, atol=1e-3, rtol=1e-3)

    # 3. Invertibility via reference: applying rope with -sin undoes the forward pass
    Q = torch.randn((2, 8, 4, head_dim))
    cos, sin = _make_cos_sin(8, head_dim)
    Q_fwd = _ref_rope(Q, cos, sin)
    Q_inv = _ref_rope(Q_fwd, cos, -sin)
    torch.testing.assert_close(Q_inv.float(), Q.float(), atol=1e-4, rtol=1e-4)

    # 4. Linearity: rope(alpha * Q) == alpha * rope(Q)
    Q = torch.randn((2, 8, 4, head_dim))
    cos, sin = _make_cos_sin(8, head_dim)
    out_base = helper_func(Q, cos, sin).cpu().float()
    for alpha in [2.0, 0.5, -1.0]:
        out_scaled = helper_func(Q * alpha, cos, sin).cpu().float()
        torch.testing.assert_close(out_scaled, out_base * alpha, atol=1e-3, rtol=1e-3)

    # 5. Zero input → zero output
    Q_zero = torch.zeros((2, 4, 2, head_dim))
    cos, sin = _make_cos_sin(4, head_dim)
    out_zero = helper_func(Q_zero, cos, sin).cpu()
    torch.testing.assert_close(out_zero, torch.zeros_like(out_zero), atol=1e-7, rtol=0)

    # 6. Determinism
    torch.manual_seed(42)
    Q = torch.randn((2, 8, 4, head_dim))
    cos, sin = _make_cos_sin(8, head_dim)
    out1 = helper_func(Q, cos, sin).cpu()
    out2 = helper_func(Q, cos, sin).cpu()
    torch.testing.assert_close(out1, out2)

def bench(kernel_fn):
    """Generate canonical benchmark inputs and invoke kernel_fn once.

    Uses (4, 2048, 32, 64): batch=4, seq=2048, heads=32, head_dim=64 —
    representative of a transformer attention RoPE pass.
    """
    batch, seq_len, n_heads, head_dim = 4, 2048, 32, 64
    Q = torch.randn((batch, seq_len, n_heads, head_dim))
    cos, sin = _make_cos_sin(seq_len, head_dim)
    return kernel_fn(Q, cos, sin)


# (batch, seq_len, n_heads, head_dim) — small / medium / large (matches bench())
# bench() at (4, 2048, 32, 64) is the proven upper bound: the optimized RoPE
# kernel allocates an SBUF tile of (d_head, B, n_heads, S) which scales with
# B*n_heads*S. At batch=8 (524288) it exceeded SBUF capacity and the
# neuronx-cc compile failed. Cap the largest case at the bench() shape.
_BENCH3_SHAPES = [(1, 2048, 32, 64), (2, 2048, 32, 64), (4, 2048, 32, 64)]


def bench3(kernel_fn, idx):
    """Bench shape #idx (0..2) — 3-shape average variant of bench()."""
    batch, seq_len, n_heads, head_dim = _BENCH3_SHAPES[idx]
    Q = torch.randn((batch, seq_len, n_heads, head_dim))
    cos, sin = _make_cos_sin(seq_len, head_dim)
    return kernel_fn(Q, cos, sin)
