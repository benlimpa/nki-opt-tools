import math
import torch

def _mx4_nibble_to_float(nibble: int, ebits: int = 2, mbits: int = 1) -> float:
    """Convert a 4-bit MX4 nibble to a floating-point value.

    Layout (MSB → LSB): [sign | exp (ebits) | mantissa (mbits)]
    """
    fp4_exp_bias = (1 << (ebits - 1)) - 1
    sign = (nibble >> (ebits + mbits)) & 1
    exp_bits = (nibble >> mbits) & ((1 << ebits) - 1)
    mantissa_bits = nibble & ((1 << mbits) - 1)
    if exp_bits == 0:
        # Subnormal: 0.mantissa * 2^(1 - bias)
        val = mantissa_bits / (1 << mbits) * (2.0 ** (1 - fp4_exp_bias))
    else:
        # Normal: (1 + mantissa/2^mbits) * 2^(exp - bias)
        val = (1.0 + mantissa_bits / (1 << mbits)) * (2.0 ** (exp_bits - fp4_exp_bias))
    return -val if sign else val


def _ref_dequantize_mx4(
    packed: torch.Tensor,
    group_size: int = 32,
    ebits: int = 2,
    mbits: int = 1,
) -> torch.Tensor:
    """Pure-PyTorch reference MX4 dequantizer for round-trip testing.

    Expects a flat (or will be flattened) uint8 tensor produced by
    ``quantize_mx4``. Returns a float32 CPU tensor.
    """
    FP32_EXP_BIAS = 127
    packed_group_size = group_size // 2 + 1
    data = packed.flatten().cpu().to(torch.int32)
    num_groups = data.numel() // packed_group_size

    values: list[float] = []
    for g in range(num_groups):
        base = g * packed_group_size
        # Exponent byte sits after the group_size // 2 packed bytes.
        exp_byte = data[base + group_size // 2].item()
        real_exp = exp_byte - FP32_EXP_BIAS
        scale = 2.0 ** real_exp
        for i in range(group_size // 2):
            b = data[base + i].item() & 0xFF
            low = _mx4_nibble_to_float(b & 0xF, ebits, mbits)
            high = _mx4_nibble_to_float((b >> 4) & 0xF, ebits, mbits)
            values.append(low * scale)
            values.append(high * scale)

    return torch.tensor(values, dtype=torch.float32)


def _expected_output_numel(M: int, K: int, group_size: int = 32) -> int:
    """Number of uint8 bytes produced by quantize_mx4 for an [M, K] input."""
    groups_per_row = math.ceil(K / group_size)
    num_groups = M * groups_per_row
    return (num_groups * group_size) // 2 + num_groups


def run_test(helper_func):
    """Property-based test for the MX4 quantize kernel.

    ``helper_func(a)`` must accept a float32 CPU tensor and return a uint8
    tensor (possibly on device). Comparisons are done after ``.cpu()``.
    """
    group_size = 32

    # 1. Output dtype is uint8
    a = torch.randn(4, 64)
    out = helper_func(a)
    assert out.dtype == torch.uint8, f"Expected uint8, got {out.dtype}"

    # 2. Output numel matches expected packed size
    M, K = 4, 64
    a = torch.randn(M, K)
    out = helper_func(a)
    expected = _expected_output_numel(M, K, group_size)
    assert out.numel() == expected, f"Expected {expected} bytes, got {out.numel()}"

    # 3. Round-trip: dequantized output is finite
    a = torch.randn(2, 32)
    out = helper_func(a)
    recovered = _ref_dequantize_mx4(out.cpu(), group_size)
    assert torch.isfinite(recovered).all(), "Dequantized output must be finite"

    # 4. Zero inputs round-trip exactly
    a = torch.zeros(1, 32)
    out = helper_func(a)
    recovered = _ref_dequantize_mx4(out.cpu(), group_size)
    assert (recovered == 0.0).all(), "Zero input must dequantize back to zero"

    # 5. Determinism (non-stochastic default mode)
    torch.manual_seed(42)
    a = torch.randn(4, 64)
    out1 = helper_func(a)
    out2 = helper_func(a)
    assert torch.equal(out1.cpu(), out2.cpu()), "Non-stochastic quantizer must be deterministic"

def bench(kernel_fn):
    """Generate canonical benchmark inputs and invoke kernel_fn once.

    Uses a (512, 2048) float32 matrix — representative of MX4 quantization of
    a transformer weight or activation block.
    """
    a = torch.randn(512, 2048)
    return kernel_fn(a)


# (M, K) — small / medium (matches bench()) / large
_BENCH3_SHAPES = [(256, 2048), (512, 2048), (1024, 2048)]


def bench3(kernel_fn, idx):
    """Bench shape #idx (0..2) — 3-shape average variant of bench()."""
    M, K = _BENCH3_SHAPES[idx]
    a = torch.randn(M, K)
    return kernel_fn(a)
