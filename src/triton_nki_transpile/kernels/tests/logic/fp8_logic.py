import torch

def _ref_per_token_group_quant_8bit(x: torch.Tensor, group_size: int, eps: float = 1e-10):
    """Pure-PyTorch reference for per-token-group quantization."""
    orig_shape = x.shape
    x_flat = x.reshape(-1, group_size).float()

    if hasattr(torch, "float8_e5m2"):
        dtype_q = torch.float8_e5m2
        info = torch.finfo(dtype_q)
    else:
        dtype_q = torch.int8
        info = torch.iinfo(dtype_q)

    bit8_max = float(info.max)
    bit8_min = float(info.min)

    absmax = x_flat.abs().max(dim=-1, keepdim=True).values
    absmax = torch.clamp(absmax, min=eps)

    scales = absmax / bit8_max
    x_q_float = torch.clamp(x_flat / scales, bit8_min, bit8_max)

    x_q = x_q_float.reshape(orig_shape).to(dtype_q)
    scales = scales.reshape(orig_shape[:-1] + (orig_shape[-1] // group_size,))

    return x_q, scales

def run_test(helper_func):
    """Property-based test for the quantization kernel."""
    # 1. Differential test
    for rows, cols, group_size in [(16, 128, 32), (8, 256, 64), (4, 512, 128)]:
        x = torch.randn((rows, cols))
        ref_q, ref_s = _ref_per_token_group_quant_8bit(x, group_size)
        act_q, act_s = helper_func(x, group_size)

        # Compare scales
        torch.testing.assert_close(act_s.cpu(), ref_s, atol=1e-5, rtol=1e-5)
        # Compare quantized values (after converting to float)
        torch.testing.assert_close(act_q.cpu().float(), ref_q.float(), atol=1e-3, rtol=1e-3)

    # 2. Determinism
    torch.manual_seed(42)
    x = torch.randn((16, 128))
    q1, s1 = helper_func(x, 32)
    q2, s2 = helper_func(x, 32)
    torch.testing.assert_close(q1.cpu().float(), q2.cpu().float())
    torch.testing.assert_close(s1.cpu(), s2.cpu())

def bench(kernel_fn):
    """Generate canonical benchmark inputs and invoke kernel_fn once.

    Uses (512, 2048) with group_size=128 — representative of per-token-group
    FP8 quantization of a transformer activation matrix.
    """
    x = torch.randn((512, 2048))
    return kernel_fn(x, 128)


# (rows, cols) — small / medium (matches bench()) / large; group_size=128
_BENCH3_SHAPES = [(256, 2048), (512, 2048), (1024, 2048)]


def bench3(kernel_fn, idx):
    """Bench shape #idx (0..2) — 3-shape average variant of bench()."""
    x = torch.randn(_BENCH3_SHAPES[idx])
    return kernel_fn(x, 128)
