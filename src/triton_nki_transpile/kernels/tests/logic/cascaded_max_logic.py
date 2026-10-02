import torch


def _ref_cascaded_max(x: torch.Tensor):
    """PyTorch reference: max value and index over last dimension, keepdim=True."""
    return torch.max(x, dim=-1, keepdim=True)


def run_test(helper_func):
    """Property-based test for the cascaded_max kernel.

    ``helper_func(x)`` must accept a CPU tensor of shape [..., V] and return
    ``(max_values, max_indices)`` tensors (possibly on device). Comparison is
    done after ``.cpu()``.

    Constraints imposed by the NKI kernel:
        - V >= 128 (reduction dimension must be at least one tile)
        - BxS = product of all dims except last must be <= 128
    """
    shapes = [
        (1, 128),    # single row, minimum vocab size
        (4, 256),    # small batch
        (8, 512),    # medium vocab
        (16, 128),   # larger batch, minimum vocab
        (1, 2, 128), # 3-D input: B=1, S=2, V=128
        (2, 4, 256), # 3-D input: B=2, S=4, V=256
    ]

    for shape in shapes:
        x = torch.randn(shape)
        ref = _ref_cascaded_max(x)
        ref_vals = ref.values
        ref_idx = ref.indices

        got_vals, got_idx = helper_func(x)
        got_vals = got_vals.cpu().float()
        got_idx = got_idx.cpu().long()

        torch.testing.assert_close(got_vals, ref_vals, atol=1e-3, rtol=1e-3,
                                   msg=f"max values mismatch for shape {shape}")
        # Indices may differ when there are ties; validate via values
        # (gather the value at the returned index and check it equals the max)
        x_flat = x.reshape(-1, x.shape[-1])
        idx_flat = got_idx.reshape(-1, 1)
        gathered = x_flat.gather(1, idx_flat).reshape(got_vals.shape)
        torch.testing.assert_close(gathered.float(), got_vals,
                                   atol=1e-3, rtol=1e-3,
                                   msg=f"index->value mismatch for shape {shape}")

    # Determinism
    x = torch.randn(8, 256)
    out1_v, out1_i = helper_func(x)
    out2_v, out2_i = helper_func(x)
    torch.testing.assert_close(out1_v.cpu(), out2_v.cpu())
    torch.testing.assert_close(out1_i.cpu(), out2_i.cpu())

    # All-same input: every index is valid (just check value is correct)
    x_const = torch.full((4, 128), 3.14)
    got_v, _ = helper_func(x_const)
    torch.testing.assert_close(got_v.cpu().float(),
                               torch.full((4, 1), 3.14), atol=1e-3, rtol=1e-3)

def bench(kernel_fn):
    """Generate canonical benchmark inputs and invoke kernel_fn once.

    Uses (128, 8192): BxS=128 (kernel constraint), vocab=8192 —
    representative of a token-level argmax over a moderate vocabulary.
    """
    x = torch.randn((128, 8192))
    return kernel_fn(x)


# (BxS, V) — small / medium (matches bench()) / large
# Constraints: BxS <= 128, V >= 128
_BENCH3_SHAPES = [(64, 8192), (128, 8192), (128, 16384)]


def bench3(kernel_fn, idx):
    """Bench shape #idx (0..2) — 3-shape average variant of bench()."""
    x = torch.randn(_BENCH3_SHAPES[idx])
    return kernel_fn(x)
